import copy
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from presidio_evaluator.evaluation import Evaluator, EvaluationResult, ModelError


class SavableFigure():

    def __init__(self, figure: Figure, file_name: str) -> None:
        self.figure = figure
        self.file_name = file_name


class ExperimentPlotter:
    """
    Plot scores (f2, precision, recall) and errors (false-positivies, false-negatives, wrong entity)
    for a PII detection model evaluated via Evaluator

    :param model_name: name of the model to be used in the plot title
    """

    def __init__(self, model_name: str):
        self.model_name = model_name.replace("/", "-")
        self._default_image_extension = "png"

    def plot_scores(self, results: EvaluationResult, beta: float = 2, output_folder=None) -> None:
        """
        Plots per-entity recall, precision, or F2 score for evaluated model.
        :param results: evaluation results
        :param beta: a float with the beta parameter of the F measure
        :param output_folder: If provided, the figures will be saved in the output folder.
        """
        scores = {}

        entity_recall_dict = copy.deepcopy(results.entity_recall_dict)
        entity_precision_dict = copy.deepcopy(results.entity_precision_dict)

        scores["entity"] = list(entity_recall_dict.keys())
        scores["recall"] = list(entity_recall_dict.values())
        scores["precision"] = list(entity_precision_dict.values())
        scores["count"] = list(results.n_dict.values())

        scores[f"f{beta}_score"] = [
            Evaluator.f_beta(precision=precision, recall=recall, beta=beta)
            for recall, precision in zip(scores["recall"], scores["precision"])
        ]

        # Add PII detection rates
        f_beta_score = f"f{beta}_score"

        scores["entity"].append("PII")
        scores["recall"].append(results.pii_recall)
        scores["precision"].append(results.pii_precision)
        scores["count"].append(results.n)
        scores[f_beta_score].append(results.pii_f)

        df = pd.DataFrame(scores)
        df["model"] = self.model_name

        f_beta_score_fig = self._plot_one_metric(df, metric=f_beta_score)
        recall_fig = self._plot_one_metric(df, metric="recall")
        precision_fig = self._plot_one_metric(df, metric="precision")
        figs = [
            SavableFigure(figure=f_beta_score_fig, file_name=f"f{beta}-score.{self._default_image_extension}"),
            SavableFigure(figure=recall_fig, file_name=f"recall.{self._default_image_extension}"),
            SavableFigure(figure=precision_fig, file_name=f"precision.{self._default_image_extension}"),
        ]

        for savable_figure in figs:
            fig = savable_figure.figure
            if output_folder:
                fig.write_image(
                    Path(output_folder, savable_figure.file_name)
                )
            else:
                fig.show()

    def plot_confusion_matrix(
            self, entities: List[str], confmatrix: List[List[int]], output_folder=None
    ) -> None:
        """
        Plots confusion matrix for evaluated model.
        :param entities: entities labels
        :param confmatrix: confusion matrix
        :param output_folder: If provided, the figure will be saved in the output folder.
        """
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
            title=f"Confusion Matrix for {self.model_name}",
            text_auto=True,
        )
        fig.update_xaxes(tickangle=90, side="top", title_standoff=10)
        fig.update_traces(textfont=dict(size=10))
        fig.update_layout(width=800, height=800)

        if output_folder:
            fig.write_image(
                Path(
                    output_folder,
                    f"confusion-matrix.{self._default_image_extension}",
                )
            )
        else:
            fig.show()

    def plot_most_common_tokens(self, results: EvaluationResult, output_folder=None) -> None:
        """
        Graph most common false positive and false negative tokens for each entity.
        :param results: evaluation results
        :param output_folder: If provided, the figures will be saved in the output folder.
        """
        fps_frames = []
        fns_frames = []

        entities = list(results.n_dict.keys())
        for entity in entities:
            fps_df = ModelError.get_fps_dataframe(
                results.model_errors, entity=entity, verbose=False
            )
            if fps_df is not None:
                fps_frames.append(fps_df)

            fns_df = ModelError.get_fns_dataframe(
                results.model_errors, entity=entity, verbose=False
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

        figs = [
            SavableFigure(figure=fn_fig, file_name=f"false-negatives.{self._default_image_extension}"),
            SavableFigure(figure=fp_fig, file_name=f"false-positives.{self._default_image_extension}")
        ]
        for savable_figure in figs:
            if not savable_figure:
                continue

            fig = savable_figure.figure
            if output_folder:
                fig.write_image(
                    Path(output_folder, savable_figure.file_name)
                )
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
