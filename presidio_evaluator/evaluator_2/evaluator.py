from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Dict
import itertools

from tqdm import tqdm

from presidio_evaluator import Span
from presidio_evaluator.evaluator_2 import (TokenOutput,
                                            SpanOutput,
                                            ModelPrediction,
                                            EvaluationResult,
                                            SampleError,
                                            evaluation_helpers)


class Evaluator:
    def __init__(
            self,
            entities_to_keep: List[str],
            compare_by_io: bool = True
    ):
        """
        Constructs all the necessary attributes for the Evaluator object

        :param entities_to_keep: List of entity names to focus the evaluator on (and ignore the rest).
        Default is None = all entities. If the provided model has a list of entities to keep,
        this list would be used for evaluation.
        :param compare_by_io: True if comparison should be done on the entity
        level and not the sub-entity level
        """
        self.compare_by_io = compare_by_io
        self.entities_to_keep = entities_to_keep

        # set up a dict for storing the span metrics
        self.span_category_output = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}
        # copy results dict to cover the four evaluation schemes for PII.
        self.span_pii_eval = {'strict': Counter(self.span_category_output),
                              'ent_type': Counter(self.span_category_output),
                              'partial': Counter(self.span_category_output),
                              'exact': Counter(self.span_category_output)}
        # copy results dict to cover the four evaluation schemes for each entity in entities_to_keep.
        self.span_entity_eval = {e: deepcopy(self.span_pii_eval) for e in self.entities_to_keep}

    def compare_token(self, annotated_tokens: List[str], predicted_tokens: List[str]) -> \
            Tuple[List[TokenOutput], Counter]:
        """
        Compares ground truth tags (annotation) and predicted (prediction) at token level.
        Return a list of TokenOutput and a list of objects of type Counter with structure {(actual, predicted) : count}

        :param annotated_tokens: truth annotation tokens from InputSample
        :param predicted_tokens: predicted tokens from PII model/system
        """
        raise NotImplementedError

    # def compare_span(self, annotated_spans: List[Span], predicted_spans: List[Span]) -> Tuple[
    #                                 List[SpanOutput], Dict[str, Counter], Dict[str, Dict[str, Counter]]]:
    @staticmethod
    def compare_span(annotated_spans: List[Span], predicted_spans: List[Span]) -> List[SpanOutput]:
        """
        Compares ground truth tags (annotation) and predicted (prediction) at span level.

        :param annotated_spans: truth annotation span from InputSample
        :param predicted_spans: predicted span from PII model/system
        :returns:
        List[SpanOutput]: a list of SpanOutput
        dict: a dictionary of global PII results with structure {eval_type : {}}
        dict: a dictionary of PII results per entity with structure {entity_name: {eval_type : {}}}
        """
        # keep track for further analysis
        span_outputs = []
        # keep track of MISS spans
        # go through each predicted
        miss_spans = deepcopy(annotated_spans)
        for prediction in predicted_spans:
            found_overlap = False
            # Scenario I: Exact match between true and prediction
            if prediction in annotated_spans:
                span_outputs.append(SpanOutput(
                    output_type="STRICT",
                    predicted_span=prediction,
                    annotated_span=prediction,
                    overlap_score=1
                ))
                found_overlap = True
                # remove this predicted span from miss_spans
                miss_spans = [x for x in miss_spans if x != prediction]
            else:
                # check overlaps with every span in true entities
                for true in annotated_spans:
                    # calculate the overlap ratio between true and pred
                    overlap_ratio = prediction.get_overlap_ratio(true, ignore_entity_type=True)
                    # Scenario IV: Offsets match, but entity type is wrong
                    if overlap_ratio == 1 and true.entity_type != prediction.entity_type:
                        span_outputs.append(SpanOutput(
                            output_type="EXACT",
                            predicted_span=prediction,
                            annotated_span=true,
                            overlap_score=1
                        ))
                        found_overlap = True
                        # remove this predicted span from miss_spans
                        miss_spans = [x for x in miss_spans if x != true]
                        break
                    # Scenario V: There is an overlap (but offsets don't match and entity type is correct)
                    elif overlap_ratio > 0 and prediction.entity_type == true.entity_type:
                        span_outputs.append(SpanOutput(
                            output_type="ENT_TYPE",
                            predicted_span=prediction,
                            annotated_span=true,
                            overlap_score=overlap_ratio
                        ))
                        found_overlap = True
                        # remove this predicted span from miss_spans
                        miss_spans = [x for x in miss_spans if x != true]
                        break
                    # Scenario VI: There is an overlap (but offsets don't match and entity type is wrong)
                    elif overlap_ratio > 0 and prediction.entity_type != true.entity_type:
                        span_outputs.append(SpanOutput(
                            output_type="PARTIAL",
                            predicted_span=prediction,
                            annotated_span=true,
                            overlap_score=overlap_ratio
                        ))
                        found_overlap = True
                        # remove this predicted span from miss_spans
                        miss_spans = [x for x in miss_spans if x != true]
                        break
            # Scenario II: No overlap with any true entity
            if not found_overlap:
                span_outputs.append(SpanOutput(
                    output_type="SPURIOUS",
                    predicted_span=prediction,
                    overlap_score=0
                ))
        # Scenario III: Span is missing in predicted list
        if len(miss_spans) > 0:
            for miss in miss_spans:
                span_outputs.append(SpanOutput(
                    output_type="MISSED",
                    annotated_span=miss,
                    overlap_score=0
                ))

        return span_outputs

    def get_span_eval_schema(self, span_outputs: List[SpanOutput]) -> Dict[str, Dict[str, Counter]]:
        """Update the evaluation schema with the new schema.
        param:span_outputs (dict): The new schema to update the evaluation schema with.
        returns: dict: The updated evaluation schema.
        """
        for span_output in span_outputs:
            if span_output.output_type == "STRICT":
                for eval_type in ["strict", "ent_type", "partial", "exact"]:
                    self.span_pii_eval[eval_type]["correct"] += 1
                    self.span_entity_eval[span_output.annotated_span.entity_type][eval_type]["correct"] += 1
            elif span_output.output_type == "EXACT":
                for eval_type in ["strict", "ent_type"]:
                    self.span_pii_eval[eval_type]["incorrect"] += 1
                    self.span_entity_eval[span_output.annotated_span.entity_type][eval_type]["incorrect"] += 1
                for eval_type in ["partial", "exact"]:
                    self.span_pii_eval[eval_type]['correct'] += 1
                    self.span_entity_eval[span_output.annotated_span.entity_type][eval_type]["correct"] += 1
            elif span_output.output_type == "ENT_TYPE":
                self.span_pii_eval["strict"]["incorrect"] += 1
                self.span_pii_eval["ent_type"]["correct"] += 1
                self.span_pii_eval["partial"]["partial"] += 1
                self.span_pii_eval["exact"]["incorrect"] += 1
                self.span_entity_eval[span_output.annotated_span.entity_type]["strict"]["incorrect"] += 1
                self.span_entity_eval[span_output.annotated_span.entity_type]["ent_type"]["correct"] += 1
                self.span_entity_eval[span_output.annotated_span.entity_type]["partial"]["partial"] += 1
                self.span_entity_eval[span_output.annotated_span.entity_type]["exact"]["incorrect"] += 1
            elif span_output.output_type == "PARTIAL":
                for eval_type in ["strict", "ent_type", "exact"]:
                    self.span_pii_eval[eval_type]['incorrect'] += 1
                    self.span_entity_eval[span_output.annotated_span.entity_type][eval_type]["incorrect"] += 1
                self.span_pii_eval["partial"]["partial"] += 1
                self.span_entity_eval[span_output.annotated_span.entity_type]["partial"]["partial"] += 1
            elif span_output.output_type == "SPURIOUS":
                for eval_type in ["strict", "ent_type", "partial", "exact"]:
                    self.span_pii_eval[eval_type]["spurious"] += 1
                    self.span_entity_eval[span_output.predicted_span.entity_type][eval_type]["spurious"] += 1
            elif span_output.output_type == "MISSED":
                for eval_type in ["strict", "ent_type", "partial", "exact"]:
                    self.span_pii_eval[eval_type]["missed"] += 1
                    self.span_entity_eval[span_output.annotated_span.entity_type][eval_type]["missed"] += 1

    def evaluate_all(self, model_predictions: List[ModelPrediction]) -> EvaluationResult:
        """
        Evaluate the PII performance at token and span levels for all sample in the reference dataset.
        :param model_predictions: list of ModelPrediction
        :returns:
        EvaluationResult: the evaluation outcomes in EvaluationResult format
        """
        sample_errors = []
        for model_prediction in tqdm(model_predictions, desc="Evaluating process...."):
            # Span evaluation
            annotated_spans = model_prediction.annotated_spans
            predicted_spans = model_prediction.predicted_spans
            span_outputs = self.get_span_outputs(annotated_spans, predicted_spans)
            # Update the evaluation schema
            self.get_span_eval_schema(span_outputs)
            sample_errors.append(SampleError(
                full_text=model_prediction.input_sample.full_text,
                metadata=model_prediction.input_sample.metadata,
                token_output=None,  # TODO: replace by output of compare_token function
                span_output=span_outputs
            ))

        # Calculate the possible and actual for the whole dataset
        self.cal_possible_actual_span_pii()
        # Calculate the precision and recall for the whole dataset
        self.cal_precision_recall_span_pii()

        return EvaluationResult(
            sample_errors=sample_errors,
            span_model_metrics=self.span_model_metrics
        )

    def cal_possible_actual_span_pii(self) -> None:
        """
        Calculate the number of actual, possible from the category errors in span_model_metrics.
        :returns:
        the self.span_model_metrics is updated with the actual and possible values
        """
        # Calculate the overall and entity level possible and actual for the whole dataset
        for entity_type, entity_level in self.span_model_metrics.items():
            for eval_type in entity_level:
                self.span_model_metrics[entity_type][eval_type] = evaluation_helpers.get_actual_possible_span(
                    self.span_model_metrics[entity_type][eval_type])

    def cal_precision_recall_span_pii(self) -> None:
        """
        Calculate the precision and recall from the category errors in span_model_metrics.
        :returns:
        the self.span_model_metrics is updated with the precision and recall values
        """
        # Calculate the overall and entity level precision and recall for the whole dataset
        for entity in self.entities_to_keep:
            self.span_model_metrics[entity] = evaluation_helpers.span_compute_precision_recall_wrapper(
                self.span_model_metrics[entity])
