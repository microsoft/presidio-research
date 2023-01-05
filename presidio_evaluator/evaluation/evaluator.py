from collections import Counter
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import numpy as np
from tqdm import tqdm
from copy import deepcopy

import plotly.express as px
import pandas as pd

from presidio_evaluator import InputSample, evaluation_helpers
from presidio_evaluator.evaluation import TokenOutput, SpanOutput, ModelPrediction, EvaluationResult, SampleError
from presidio_evaluator.models import BaseModel


class Evaluator:
    def __init__(
        self,
        verbose: bool = False,
        compare_by_io=True,
        entities_to_keep: Optional[List[str]] = None,
        span_overlap_threshold: float = 0.5
    ):
        """
        Evaluate a PII detection model or a Presidio analyzer / recognizer

        :param compare_by_io: True if comparison should be done on the entity
        level and not the sub-entity level
        :param entities_to_keep: List of entity names to focus the evaluator on (and ignore the rest).
        Default is None = all entities. If the provided model has a list of entities to keep,
        this list would be used for evaluation.
        """
        self.verbose = verbose
        self.compare_by_io = compare_by_io
        self.entities_to_keep = entities_to_keep
        self.span_overlap_threshold = span_overlap_threshold


    def compare_token(self, model_prediction: ModelPrediction) -> Tuple[List[TokenOutput], Counter]:
        """
        Compares ground truth tags (annotation) and predicted (prediction) at token level. 
        Return a list of TokenOutput and a list of objects of type Counter with structure {(actual, predicted) : count}
        :param model_prediction: model_prediction containing an InputSample and a list of predicted tags and tokens
        """
        annotation = model_prediction.input_sample.tags
        tokens = model_prediction.input_sample.tokens

        if len(annotation) != len(prediction):
            print(
                "Annotation and prediction do not have the"
                "same length. Sample={}".format(model_prediction.input_sample)
            )
            return Counter(), []

        results = Counter()
        token_errors = []

        new_annotation = annotation.copy()

        if self.compare_by_io:
            new_annotation = self._to_io(new_annotation)
            prediction = self._to_io(model_prediction.predicted_tags)

        # Ignore annotations that aren't in the list of
        # requested entities.
        if self.entities_to_keep:
            prediction = self._adjust_per_entities(prediction)
            new_annotation = self._adjust_per_entities(new_annotation)
        for i in range(0, len(new_annotation)):
            results[(new_annotation[i], prediction[i])] += 1

            if self.verbose:
                print("Annotation:", new_annotation[i])
                print("Prediction:", prediction[i])
                print(results)

            # check if there was an error
            is_error = new_annotation[i] != prediction[i]
            if is_error:
                if prediction[i] == "O":
                    token_errors.append(
                        TokenOutput(
                            error_type="FN",
                            annotation=new_annotation[i],
                            prediction=prediction[i],
                            token=tokens[i],
                        )
                    )
                elif new_annotation[i] == "O":
                    token_errors.append(
                        TokenOutput(
                            error_type="FP",
                            annotation=new_annotation[i],
                            prediction=prediction[i],
                            token=tokens[i],
                        )
                    )
                else:
                    token_errors.append(
                        TokenOutput(
                            error_type="Wrong entity",
                            annotation=new_annotation[i],
                            prediction=prediction[i],
                            token=tokens[i],
                        )
                    )

        return token_errors, results


    def compare_span(self, model_prediction: ModelPrediction) -> Tuple[List[SpanOutput], dict[dict]]:
        """
        Compares ground truth tags (annotation) and predicted (prediction) at span level. 
        :param model_prediction: model_prediction containing an InputSample and a list of predicted tags and tokens
        Returns:
        List[SpanOutput]: a list of SpanOutput
        dict: a dictionary of PII results per entity with structure {{entity_name: {output_type : count}}}
        """

        evaluation = {"strict": 0, "exact": 0, "partial": 0, "incorrect": 0, "miss": 0, "spurious": 0}
        evaluate_results = {e: deepcopy(evaluation) for e in self.entities_to_keep}

        gold_spans = model_prediction.input_sample.spans
        pred_spans = model_prediction.predicted_spans

        # keep track of true entities that overlapped
        true_which_overlapped_with_pred = []
        span_outputs = []

        for pred in pred_spans:
            model_output = evaluation_helpers.get_matched_gold(pred, gold_spans, self.span_overlap_threshold)
            output_type = model_output.output_type
            span_outputs.append(model_output)

            evaluation[output_type] += 1
            evaluate_results[model_output.gold_span.entity_type] += 1
            
            if output_type in ['strict', 'exact', 'partial', 'incorrect']:
                true_which_overlapped_with_pred.append(model_output.gold_span)
            
        ## Get all missed span/entity in the gold corpus
        for true in gold_spans:
            if true in true_which_overlapped_with_pred:
                continue
            else:
                evaluation["miss"] += 1
                evaluate_results[true.entity_type]["miss"] += 1
                # Add the output's detail to evaluation_results
                span_outputs.append(SpanOutput(
                        output_type = "miss",
                        gold_span = true,
                        overlap_score=0
                    ))
        evaluate_results['PII'] = evaluation
        return span_outputs, evaluate_results

    def evaluate_all(self, model_predictions: List[ModelPrediction]) -> EvaluationResult:
        """
        Evaluate the PII performance at token and span levels. 
        :param model_predictions: list of ModelPrediction
        Returns:
        EvaluationResult: the evaluation outcomes in EvaluationResult format
        """
        sample_errors = []
        # Hold the span PII results for the whole dataset with structure {{entity_name: {output_type : count}}}
        all_span_evaluate_results = {}
        # Hold the token PII results for the whole dataset with structure {(actual, predicted) : count}
        all_token_evaluate_results = Counter()

        for model_prediction in model_predictions:
            span_outputs, span_evaluate_results = self.compare_span(model_prediction)
            token_errors, token_evaluate_results = self.compare_token(model_prediction)
            sample_errors.append(SampleError(
                span_outputs=span_outputs,
                token_outputs = token_errors,
                full_text=model_prediction.input_sample.full_text,
                metadata=model_prediction.input_sample.metadata
            ))
            # add span_evaluate_results to all_span_evaluate_results
            all_span_evaluate_results = evaluation_helpers.dict_merge(all_span_evaluate_results, span_evaluate_results)
            # add token_evaluate_results to all_token_evaluate_results
            all_token_evaluate_results += token_evaluate_results

        ## Calculate the metrics from the evaluated results
        span_model_metrics = {}
        # For span evaluation
        for entity_type in all_span_evaluate_results:
            # Compute actual and possible for each entity and PII
            span_model_metrics["span_distribution"][entity_type] = evaluation_helpers.span_compute_actual_possible(all_span_evaluate_results[entity_type])
            # Calculate precision, recall for each entity and PII 
            span_model_metrics["metrics"][entity_type] = evaluation_helpers.span_compute_precision_recall(span_model_metrics["span_distribution"][entity_type])

        # For token evaluation
        token_model_metrics = evaluation_helpers.token_calulate_score(all_token_evaluate_results)

        return EvaluationResult(
            sample_errors = sample_errors,
            token_confusion_matrix = all_token_evaluate_results,
            token_model_metrics = token_model_metrics,
            span_model_metrics = span_model_metrics
        )
            
    # TODO: Review and refactor (if needed) the functions below 
    def _adjust_per_entities(self, tags):
        if self.entities_to_keep:
            return [tag if tag in self.entities_to_keep else "O" for tag in tags]
        else:
            return tags

    @staticmethod
    def _to_io(tags):
        """
        Translates BILUO/BIO/IOB to IO - only In or Out of entity.
        ['B-PERSON','I-PERSON','L-PERSON'] is translated into
        ['PERSON','PERSON','PERSON']
        :param tags: the input tags in BILUO/IOB/BIO format
        :return: a new list of IO tags
        """
        return [tag[2:] if "-" in tag else tag for tag in tags]

    def evaluate_sample(
        self, sample: InputSample, prediction: List[str]
    ) -> EvaluationResult:
        if self.verbose:
            print("Input sentence: {}".format(sample.full_text))

        results, mistakes = self.compare(input_sample=sample, prediction=prediction)
        return EvaluationResult(results, mistakes, sample.full_text)

    def evaluate_all(self, dataset: List[InputSample]) -> List[EvaluationResult]:
        evaluation_results = []
        if self.model.entity_mapping:
            print(
                f"Mapping entity values using this dictionary: {self.model.entity_mapping}"
            )
        for sample in tqdm(dataset, desc=f"Evaluating {self.model.__class__}"):

            # Align tag values to the ones expected by the model
            self.model.align_entity_types(sample)

            # Predict
            prediction = self.model.predict(sample)

            # Remove entities not requested
            prediction = self.model.filter_tags_in_supported_entities(prediction)

            # Switch to requested labeling scheme (IO/BIO/BILUO)
            prediction = self.model.to_scheme(prediction)

            evaluation_result = self.evaluate_sample(
                sample=sample, prediction=prediction
            )
            evaluation_results.append(evaluation_result)

        return evaluation_results

    @staticmethod
    def align_entity_types(
        input_samples: List[InputSample],
        entities_mapping: Dict[str, str] = None,
        allow_missing_mappings: bool = False,
    ) -> List[InputSample]:
        """
        Change input samples to conform with Presidio's entities
        :return: new list of InputSample
        """

        new_input_samples = input_samples.copy()

        # A list that will contain updated input samples,
        new_list = []

        for input_sample in new_input_samples:
            contains_field_in_mapping = False
            new_spans = []
            # Update spans to match the entity types in the values of entities_mapping
            for span in input_sample.spans:
                if span.entity_type in entities_mapping.keys():
                    new_name = entities_mapping.get(span.entity_type)
                    span.entity_type = new_name
                    contains_field_in_mapping = True

                    new_spans.append(span)
                else:
                    if not allow_missing_mappings:
                        raise ValueError(
                            f"Key {span.entity_type} cannot be found in the provided entities_mapping"
                        )
            input_sample.spans = new_spans

            # Update tags in case this sample has relevant entities for evaluation
            if contains_field_in_mapping:
                for i, tag in enumerate(input_sample.tags):
                    has_prefix = "-" in tag
                    if has_prefix:
                        prefix = tag[:2]
                        clean = tag[2:]
                    else:
                        prefix = ""
                        clean = tag

                    if clean in entities_mapping.keys():
                        new_name = entities_mapping.get(clean)
                        input_sample.tags[i] = "{}{}".format(prefix, new_name)
                    else:
                        input_sample.tags[i] = "O"

            new_list.append(input_sample)

        return new_list


    class Plotter:
        """
        Plot scores (f2, precision, recall) and errors (false-positivies, false-negatives) 
        for a PII detection model evaluated via Evaluator

        :param model: Instance of a fitted model (of base type BaseModel)
        :param results: results given by evaluator.calculate_score(evaluation_results)
        :param output_folder: folder to store plots and errors in
        :param model_name: name of the model to be used in the plot title
        :param beta: a float with the beta parameter of the F measure,
        which gives more or less weight to precision vs. recall
        """

        def __init__(self, model, results, output_folder: Path, model_name: str, beta: float):
            self.model = model
            self.results = results
            self.output_folder = output_folder
            self.model_name = model_name.replace("/", "-")
            self.errors = results.model_errors
            self.beta = beta

        def plot_scores(self) -> None:
            """
            Plots per-entity recall, precision, or F2 score for evaluated model. 
            :param plot_type: which metric to graph (default is F2 score)
            """
            scores = {}
            scores['entity'] = list(self.results.entity_recall_dict.keys())
            scores['recall'] = list(self.results.entity_recall_dict.values())
            scores['precision'] = list(self.results.entity_precision_dict.values())
            scores['count'] = list(self.results.n_dict.values())
            scores[f"f{self.beta}_score"] = [Evaluator.f_beta(precision=precision, recall=recall, beta=self.beta)
                                  for recall, precision in zip(scores['recall'], scores['precision'])]
            df = pd.DataFrame(scores)
            df['model'] = self.model_name
            self._plot(df, plot_type="f2_score")
            self._plot(df, plot_type="precision")
            self._plot(df, plot_type="recall")

        def _plot(self, df, plot_type) -> None:
            fig = px.bar(df, text_auto=".2", y='entity', orientation="h",
                         x=plot_type, color='count', barmode='group', title=f"Per-entity {plot_type} for {self.model_name}")
            fig.update_layout(barmode='group', yaxis={
                'categoryorder': 'total ascending'})
            fig.update_layout(yaxis_title=f"{plot_type}", xaxis_title="PII Entity")
            fig.update_traces(textfont_size=12, textangle=0,
                              textposition="outside", cliponaxis=False)
            fig.update_layout(
                plot_bgcolor="#FFF",
                xaxis=dict(
                    title="PII entity",
                    linecolor="#BCCCDC",  # Sets color of X-axis line
                    showgrid=False  # Removes X-axis grid lines
                ),
                yaxis=dict(
                    title=f"{plot_type}",
                    linecolor="#BCCCDC",  # Sets color of X-axis line
                    showgrid=False  # Removes X-axis grid lines
                ),
            )
            fig.show()

        def plot_most_common_tokens(self) -> None:
            """Graph most common false positive and false negative tokens for each entity."""
            ModelError.most_common_fp_tokens(self.errors)
            fps_frames = []
            fns_frames = []
            for entity in self.model.entity_mapping.values():
                fps_df = ModelError.get_fps_dataframe(self.errors, entity=[entity])
                if fps_df is not None:
                    fps_path = self.output_folder / \
                        f"{self.model_name}-{entity}-fps.csv"
                    fps_df.to_csv(fps_path)
                    fps_frames.append(fps_path)
                fns_df = ModelError.get_fns_dataframe(self.errors, entity=[entity])
                if fns_df is not None:
                    fns_path = self.output_folder / \
                        f"{self.model_name}-{entity}-fns.csv"
                    fns_df.to_csv(fns_path)
                    fns_frames.append(fns_path)

            def group_tokens(df):
                return df.groupby(['token', 'annotation']).size().to_frame(
                ).sort_values([0], ascending=False).head(3).reset_index()

            fps_tokens_df = pd.concat(
                [group_tokens(pd.read_csv(df_path)) for df_path in fps_frames])
            fns_tokens_df = pd.concat(
                [group_tokens(pd.read_csv(df_path)) for df_path in fns_frames])

            def generate_graph(title, tokens_df):
                fig = px.histogram(tokens_df, x=0, y="token", orientation='h', color='annotation',
                                   title=f"Most common {title} for {self.model_name}")

                fig.update_layout(yaxis_title=f"count", xaxis_title="PII Entity")
                fig.update_traces(textfont_size=12, textangle=0,
                                  textposition="outside", cliponaxis=False)
                fig.update_layout(
                    plot_bgcolor="#FFF",
                    xaxis=dict(
                        title="Count",
                        linecolor="#BCCCDC",  # Sets color of X-axis line
                        showgrid=False  # Removes X-axis grid lines
                    ),
                    yaxis=dict(
                        title=f"Tokens",
                        linecolor="#BCCCDC",  # Sets color of X-axis line
                        showgrid=False  # Removes X-axis grid lines
                    ),
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig.show()
            generate_graph(title="false-negatives", tokens_df=fns_tokens_df)
            generate_graph(title="false-positives", tokens_df=fps_tokens_df)
