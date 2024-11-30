import copy
from pathlib import Path
from typing import Optional, List

import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from presidio_evaluator.evaluation import Evaluator, EvaluationResult, ModelError


class Plotter:
    """
    Plot scores (f2, precision, recall) and errors
    (false-positivies, false-negatives, wrong entity)
    for a PII detection model evaluated via Evaluator

    :param results: results given by evaluator.calculate_score(evaluation_results)
    :param output_folder: folder to store plots and errors in
    :param model_name: name of the model to be used in the plot title
    :param beta: a float with the beta parameter of the F measure,
    which gives more or less weight to precision vs. recall
    :param save_as: Optional string with the path to save the plots as images or svg
    """

    def __init__(
        self,
        results: EvaluationResult,
        output_folder: Optional[Path] = None,
        model_name: str = "PresidioAnalyzerWrapper",
        beta: float = 2,
        save_as: Optional[str] = None,
    ):
        self.results = results
        self.output_folder = output_folder
        self.save_as = save_as
        self.model_name = model_name.replace("/", "-")
        self.beta = beta

        if not output_folder:
            self.output_folder = Path("plots")
        else:
            self.output_folder = Path(output_folder)

        if save_as:
            self.output_folder.mkdir(parents=True, exist_ok=True)

    def plot_scores(self) -> None:
        """
        Plots per-entity recall, precision, or F2 score for evaluated model.

        """
        scores = {}

        entity_recall_dict = copy.deepcopy(self.results.entity_recall_dict)
        entity_precision_dict = copy.deepcopy(self.results.entity_precision_dict)

        scores["entity"] = list(entity_recall_dict.keys())
        scores["recall"] = list(entity_recall_dict.values())
        scores["precision"] = list(entity_precision_dict.values())
        scores["count"] = list(self.results.n_dict.values())

        scores[f"f{self.beta}_score"] = [
            Evaluator.f_beta(precision=precision, recall=recall, beta=self.beta)
            for recall, precision in zip(scores["recall"], scores["precision"])
        ]

        # Add PII detection rates
        f_beta_score = f"f{self.beta}_score"

        scores["entity"].append("PII")
        scores["recall"].append(self.results.pii_recall)
        scores["precision"].append(self.results.pii_precision)
        scores["count"].append(self.results.n)
        scores[f_beta_score].append(self.results.pii_f)

        df = pd.DataFrame(scores)
        df["model"] = self.model_name

        beta_fig = self._plot_one_metric(df, metric=f_beta_score)
        recall_fig = self._plot_one_metric(df, metric="recall")
        precision_fig = self._plot_one_metric(df, metric="precision")
        figs = [beta_fig, recall_fig, precision_fig]

        for fig in figs:
            if self.save_as:
                fig.write_image(
                    Path(self.output_folder, f"{self.model_name}-scores.{self.save_as}")
                )
                fig.show(self.save_as)
            else:
                fig.show()

    def _plot_one_metric(self, df: pd.DataFrame, metric: str) -> Figure:
        fig = px.bar(
            df,
            text_auto=".2",
            y="entity",
            orientation="h",
            x=metric,
            color="count",
            barmode="group",
            height=max(20 * len(set(df["entity"])), 500),
            title=f"Per-entity {metric} for {self.model_name}",
        )

        subtitle = (
            f"Entity {metric} values also consider mismatches between "
            f"different entity types, not just misclassified PII values\n"
        )

        # Add a subtitle using annotations
        fig.update_layout(
            annotations=[
                dict(
                    text=subtitle if metric == "precision" else "",
                    xref="paper",
                    yref="paper",  # Use "paper" coordinates
                    x=0.5,
                    y=1.1,  # Position above the plot (centered)
                    showarrow=False,
                    font=dict(size=12, color="gray"),
                )
            ]
        )

        fig.update_layout(barmode="group", yaxis={"categoryorder": "total ascending"})
        fig.update_layout(yaxis_title=f"{metric}", xaxis_title="PII Entity")
        fig.update_traces(
            textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
        )
        fig.update_layout(
            plot_bgcolor="#FFF",
            xaxis=dict(
                title="PII entity",
                linecolor="#BCCCDC",  # Sets color of X-axis line
                showgrid=False,  # Removes X-axis grid lines
            ),
            yaxis=dict(
                title=f"{metric}",
                linecolor="#BCCCDC",  # Sets color of X-axis line
                showgrid=False,  # Removes X-axis grid lines
            ),
        )
        return fig

    def plot_most_common_tokens(self) -> None:
        """Graph most common false positive and false negative tokens for each entity."""
        fps_frames = []
        fns_frames = []

        entities = list(self.results.n_dict.keys())
        for entity in entities:
            fps_df = ModelError.get_fps_dataframe(
                self.results.model_errors, entity=entity, verbose=False
            )
            if fps_df is not None:
                fps_frames.append(fps_df)

            fns_df = ModelError.get_fns_dataframe(
                self.results.model_errors, entity=entity, verbose=False
            )
            if fns_df is not None:
                fns_frames.append(fns_df)

        to_concat = []
        for fps_df in fps_frames:
            grouped = self._group_tokens(fps_df, key="prediction")
            if len(grouped) > 0:
                to_concat.append(grouped)
        fps_tokens_df = pd.concat(to_concat) if to_concat else pd.DataFrame()

        for fns_df in fns_frames:
            grouped = self._group_tokens(fns_df, key="annotation")
            if len(grouped) > 0:
                to_concat.append(grouped)

        fns_tokens_df = pd.concat(to_concat) if to_concat else pd.DataFrame()

        if len(fns_tokens_df) > 0:
            fn_fig = self._plot_histogram(
                title="false-negatives", tokens_df=fns_tokens_df, key="annotation"
            )
        else:
            fn_fig = None
            print("No false negatives found")
        if len(fps_tokens_df) > 0:
            fp_fig = self._plot_histogram(
                title="false-positives",
                tokens_df=fps_tokens_df,
                key="prediction",
            )
        else:
            fp_fig = None
            print("No false positives found")

        for fig in [fn_fig, fp_fig]:
            if not fig:
                continue

            if self.save_as:
                fig.show(self.save_as)
                fig.write_image(
                    Path(
                        self.output_folder,
                        f"{self.model_name}-common-errors.{self.save_as}",
                    )
                )
            else:
                fig.show()

    @staticmethod
    def _group_tokens(df, key: str = "annotation"):
        return (
            df.groupby(["token", key])
            .size()
            .to_frame()
            .sort_values([0], ascending=False)
            .head(5)
            .reset_index()
        )

    @staticmethod
    def _plot_histogram(
        title: str,
        tokens_df: pd.DataFrame,
        key="annotation",
    ) -> Figure:
        fg = px.histogram(
            tokens_df,
            x=0,
            y="token",
            orientation="h",
            color=key,
            text_auto=True,
            title=f"Most common {title} tokens",
            height=max([30 * len(tokens_df), 500]),
        )

        fg.update_layout(yaxis_title="count", xaxis_title="PII Entity")
        fg.update_traces(
            textfont_size=8,
            textangle=0,
            textposition="outside",
            cliponaxis=True,
        )
        fg.update_layout(
            plot_bgcolor="#FFF",
            xaxis=dict(
                title="Count",
                linecolor="#BCCCDC",  # Sets color of X-axis line
                showgrid=False,  # Removes X-axis grid lines
            ),
            yaxis=dict(
                title="Tokens",
                linecolor="#BCCCDC",  # Sets color of X-axis line
                showgrid=False,  # Removes X-axis grid lines
            ),
        )
        fg.update_layout(yaxis={"categoryorder": "total ascending"})

        return fg

    def plot_confusion_matrix(
        self, entities: List[str], confmatrix: List[List[int]]
    ) -> None:
        # Create a DataFrame from the 2D list
        confusion_matrix_df = pd.DataFrame(confmatrix, index=entities, columns=entities)

        confusion_matrix_df.loc["Total"] = confusion_matrix_df.sum()

        # Add a column for the totals
        confusion_matrix_df["Total"] = confusion_matrix_df.sum(axis=1)

        # Create the heatmap
        fig = px.imshow(
            confusion_matrix_df,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=confusion_matrix_df.columns,
            y=confusion_matrix_df.index,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix for model {self.model_name}",
            text_auto=True,
        )
        fig.update_xaxes(tickangle=90, side="top", title_standoff=10)
        fig.update_traces(textfont=dict(size=10))
        fig.update_layout(width=800, height=800)

        if self.save_as:
            fig.show(self.save_as)
            fig.write_image(
                Path(
                    self.output_folder,
                    f"{self.model_name}-confusion-matrix.{self.save_as}",
                )
            )
        else:
            fig.show()
