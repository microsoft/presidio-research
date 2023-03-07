from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Dict

from tqdm import tqdm

from presidio_evaluator import Span
from presidio_evaluator.evaluator_2 import (TokenOutput,
                                            SpanOutput,
                                            ModelPrediction,
                                            SampleError,
                                            EvaluationResult)


class Evaluator:
    def __init__(
            self,
            entities_to_keep: List[str],
            compare_by_io: bool = True,
            entity_mapping: Dict[str, str] = None
    ):
        """
        Evaluate a PII detection model or a PII model.
        :param entities_to_keep: List of entity names to focus the evaluator on
        (and ignore the rest).
        Default is None = all entities. If the provided model has a list of
        entities to keep, this list would be used for evaluation.
        :param compare_by_io: True if comparison should be done on the entity
        level and not the sub-entity level
        :param entity_mapping: A dictionary of entity names to map between
        input dataset and the entities from PII model
        """
        self.compare_by_io = compare_by_io
        self.entities_to_keep = entities_to_keep
        self.entity_mapping = entity_mapping

    def compare_token(self, annotated_tokens: List[str],
                      predicted_tokens: List[str]) -> Tuple[List[TokenOutput], Counter]:
        """
        Compares ground truth tags (annotation) and predicted (prediction)
        at token level.
        :param annotated_tokens: truth annotation tokens from InputSample
        :param predicted_tokens: predicted tokens from PII model/system
        :returns: a list of TokenOutput and a list of objects of type Counter
        with structure {(actual, predicted) : count}
        """
        raise NotImplementedError

    @staticmethod
    def compare_span(
            annotated_spans: List[Span], predicted_spans: List[Span]
    ) -> List[SpanOutput]:
        """
        Compares ground truth tags (annotation) and predicted (prediction) at span level
        by using the SEM EVAL 2013 approach:
        https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        The entity mapping (e.g. PERSON->PER) should be done before calling this
        function.
        :param annotated_spans: truth annotation span from InputSample
        :param predicted_spans: predicted span from PII model/system
        :returns:
        List[SpanOutput]: a list of SpanOutput
        """
        # keep track for further analysis.
        span_outputs = []
        # keep track of MISS spans
        miss_spans = deepcopy(annotated_spans)
        # go through each predicted
        # Mapping between annotated and predicted spans
        for prediction in predicted_spans:
            found_overlap = False
            # Scenario I: Exact match between true and prediction
            if prediction in annotated_spans:
                span_outputs.append(
                    SpanOutput(
                        output_type="STRICT",
                        predicted_span=prediction,
                        annotated_span=prediction,
                        overlap_score=1,
                    )
                )
                found_overlap = True
                # remove this predicted span from miss_spans
                miss_spans = [x for x in miss_spans if x != prediction]
            else:
                # check overlaps with every span in true entities
                for true in annotated_spans:
                    # calculate the overlap ratio between true and pred
                    overlap_ratio = prediction.get_overlap_ratio(
                        true, ignore_entity_type=True
                    )
                    # Scenario IV: Offsets match, but entity type is wrong
                    if (
                            overlap_ratio == 1
                            and true.entity_type != prediction.entity_type
                    ):
                        span_outputs.append(
                            SpanOutput(
                                output_type="EXACT",
                                predicted_span=prediction,
                                annotated_span=true,
                                overlap_score=1,
                            )
                        )
                        found_overlap = True
                        # remove this predicted span from miss_spans
                        miss_spans = [x for x in miss_spans if x != true]
                        break
                    # Scenario V: There is an overlap (but offsets don't match
                    # and entity type is correct)
                    elif overlap_ratio > 0 \
                            and prediction.entity_type == true.entity_type:
                        span_outputs.append(
                            SpanOutput(
                                output_type="ENT_TYPE",
                                predicted_span=prediction,
                                annotated_span=true,
                                overlap_score=overlap_ratio,
                            )
                        )
                        found_overlap = True
                        # remove this predicted span from miss_spans
                        miss_spans = [x for x in miss_spans if x != true]
                        break
                    # Scenario VI: There is an overlap (but offsets don't match
                    # and entity type is wrong)
                    elif overlap_ratio > 0 \
                            and prediction.entity_type != true.entity_type:
                        span_outputs.append(
                            SpanOutput(
                                output_type="PARTIAL",
                                predicted_span=prediction,
                                annotated_span=true,
                                overlap_score=overlap_ratio,
                            )
                        )
                        found_overlap = True
                        # remove this predicted span from miss_spans
                        miss_spans = [x for x in miss_spans if x != true]
                        break
            # Scenario II: No overlap with any true entity
            if not found_overlap:
                span_outputs.append(
                    SpanOutput(
                        output_type="SPURIOUS",
                        predicted_span=prediction,
                        overlap_score=0,
                    )
                )
        # Scenario III: Span is missing in predicted list
        if not miss_spans:
            for miss in miss_spans:
                span_outputs.append(
                    SpanOutput(
                        output_type="MISSED", annotated_span=miss, overlap_score=0
                    )
                )

        return span_outputs

    def evaluate_all(
            self, model_predictions: List[ModelPrediction]
    ) -> EvaluationResult:
        """
        Evaluate the PII performance at token and span levels for all sample
        in the reference dataset.
        :param model_predictions: list of ModelPrediction
        :returns:
        EvaluationResult: the evaluation outcomes in EvaluationResult format
        """
        sample_errors = []
        for model_prediction in tqdm(model_predictions, desc="Evaluating process...."):
            if self.entity_mapping:
                # Align tag values to the ones expected by the model
                model_prediction.input_sample.translate_input_sample_tags(
                    dictionary=self.entity_mapping
                )
            annotated_spans = model_prediction.input_sample.spans
            predicted_spans = model_prediction.predicted_spans
            span_outputs = self.compare_span(annotated_spans, predicted_spans)
            # filter span_outputs based on entities_to_keep
            if self.entities_to_keep:
                span_outputs = self.filter_span_outputs_in_entities_to_keep(
                    span_outputs
                )
            # TODO: token evaluation
            sample_errors.append(
                SampleError(
                    full_text=model_prediction.input_sample.full_text,
                    metadata=model_prediction.input_sample.metadata,
                    token_output=None,
                    # TODO: replace by output of compare_token function
                    span_output=span_outputs
                )
            )

        return EvaluationResult(
            sample_errors=sample_errors,
            entities_to_keep=self.entities_to_keep,
        )

    def filter_span_outputs_in_entities_to_keep(self,
                                                span_outputs: List[SpanOutput]) -> \
            List[SpanOutput]:
        """
        Filter span_outputs based on the entities_to_keep list.
        :param span_outputs: list of SpanOutput
        :returns:
        List[SpanOutput]: filtered list of SpanOutput
        """
        ent_to_keep = self.entities_to_keep
        if ent_to_keep is None:
            return span_outputs
        else:
            filtered_span_outputs = []
            for span_output in span_outputs:
                if span_output.output_type in ["STRICT", "EXACT", "ENT_TYPE",
                                               "PARTIAL"]:
                    if span_output.predicted_span.entity_type in ent_to_keep \
                            or span_output.annotated_span.entity_type in ent_to_keep:
                        filtered_span_outputs.append(span_output)
                elif span_output.output_type == "SPURIOUS" and \
                        span_output.predicted_span.entity_type in ent_to_keep:
                    filtered_span_outputs.append(span_output)
                elif span_output.output_type == "MISSED" and \
                        span_output.annotated_span.entity_type in ent_to_keep:
                    filtered_span_outputs.append(span_output)
            return filtered_span_outputs
