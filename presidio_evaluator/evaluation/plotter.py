import copy
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from presidio_evaluator.evaluation import BaseEvaluator, EvaluationResult, ModelError


class Plotter:
    """
    A class for visualizing evaluation results of a PII detection model.

    The `Plotter` class provides methods to generate various plots that help analyze
    the performance of a PII detection model evaluated using the `Evaluator` class.
    These visualizations include per-entity scores, most common errors, and confusion matrices.

    Key Features:
    - Plot per-entity precision, recall, and F-beta scores.
    - Visualize the most common false positive and false negative tokens.
    - Generate a confusion matrix for entity predictions.

    Attributes:
        results (EvaluationResult): The evaluation results from the `Evaluator`.
        model_name (str): The name of the model, used in plot titles.
        beta (float): The beta parameter for the F-beta score, controlling the
                      weight of precision vs. recall.
        save_as (Optional[str]): The file format to save plots (e.g., "png", "svg").
                                 If specified, plots are saved in the given format.
                                 It has to be specified if output_folder is passed
                                 as input to the plotting functions.

    Notes:
        - plots are always displayed interactively using the default
          Plotly viewer, regardless of the `save_as` value.
        - The `output_folder` is created automatically if it does not exist.
    """

    def __init__(
        self,
        results: EvaluationResult,
        model_name: str = "PresidioAnalyzerWrapper",
        beta: float = 2,
        save_as: Optional[str] = None,
    ):
        self.results = results
        self.save_as = save_as
        self.model_name = model_name.replace("/", "-")
        self.beta = beta

    def plot_scores(self, output_folder: Optional[Union[Path, str]] = None) -> None:
        """
        Plots per-entity recall, precision, or F2 score for evaluated model.
        Parameters:
            output_folder (Path): The folder where the plots will be saved.
        """
        scores = {}

        entity_recall_dict = copy.deepcopy(self.results.entity_recall_dict)
        entity_precision_dict = copy.deepcopy(self.results.entity_precision_dict)

        scores["entity"] = list(entity_recall_dict.keys())
        scores["recall"] = list(entity_recall_dict.values())
        scores["precision"] = list(entity_precision_dict.values())
        scores["count"] = list(self.results.n_dict.values())

        scores[f"f{self.beta}_score"] = [
            BaseEvaluator.f_beta(precision=precision, recall=recall, beta=self.beta)
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
        fig_names = ["f_beta", "recall", "precision"]

        for fig, file_name in zip(figs, fig_names):
            if output_folder is not None:
                self.save_fig_to_file(
                    fig=fig,
                    output_folder=output_folder,
                    file_name=f"scores-{file_name}",
                )
                fig.show()
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

    def plot_most_common_tokens(
        self, output_folder: Optional[Union[Path, str]] = None
    ) -> None:
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

        fig_names = ["fn", "fp"]
        for fig, fig_name in zip([fn_fig, fp_fig], fig_names):
            if not fig:
                continue

            if output_folder is not None:
                self.save_fig_to_file(
                    fig=fig,
                    output_folder=output_folder,
                    file_name=f"common-errors-{fig_name}",
                )
                fig.show()
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
        self,
        entities: List[str],
        confmatrix: List[List[int]],
        output_folder: Optional[Union[Path, str]] = None,
    ) -> None:
        """
        Plot the confusion matrix for the evaluated model.
        Parameters:
            entities (List[str]): List of entity names.
            confmatrix (List[List[int]]): 2D list representing the confusion matrix.
            output_folder (Path): The folder where the plot will be saved.

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
            title=f"Confusion Matrix for model {self.model_name}",
            text_auto=True,
        )
        fig.update_xaxes(tickangle=90, side="top", title_standoff=10)
        fig.update_traces(textfont=dict(size=10))
        fig.update_layout(width=800, height=800)

        if output_folder is not None:
            self.save_fig_to_file(
                fig=fig, output_folder=output_folder, file_name="confusion-matrix"
            )
            fig.show()
        else:
            fig.show()

    def save_fig_to_file(
        self,
        fig: Figure,
        output_folder: Union[Path, str],
        file_name: str = "figure",
    ) -> None:
        """
        Save the figure to a file.
        Parameters:
            fig (Figure): The figure to save.
            output_folder (Path): The folder where the plot will be saved.
            file_name (str): The name of the file to save the plot as.
        """
        if not output_folder:
            raise ValueError("output_folder is missing, cannot save figure."
                             "If you do not wish to save figures, "
                             "configure the Plotter with `save_as = None`")

        output_folder = Path(output_folder)

        output_folder.mkdir(parents=True, exist_ok=True)
        if self.save_as == "html":
            fig.write_html(
                Path(output_folder, f"{self.model_name}-{file_name}.{self.save_as}")
            )
        elif self.save_as is not None:
            fig.write_image(
                Path(output_folder, f"{self.model_name}-{file_name}.{self.save_as}")
            )
        else:
            raise ValueError(
                "save_as must be either 'html' or a valid image format (e.g., 'png', 'svg')."
            )
