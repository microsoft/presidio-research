from collections import Counter
from typing import List, Tuple, Dict

from presidio_evaluator import Span
from presidio_evaluator.evaluator_2 import (TokenOutput,
                                            SpanOutput,
                                            ModelPrediction,
                                            EvaluationResult)


class Evaluator:
    def __init__(
            self,
            entities_to_keep: List[str],
            compare_by_io: bool = True,
            entity_mapping: Dict[str, str] = None
    ):
        """
        Constructs all the necessary attributes for the Evaluator object
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
        :param annotated_spans: truth annotation span from InputSample
        :param predicted_spans: predicted span from PII model/system
        :returns:
        List[SpanOutput]: a list of SpanOutput
        """
        raise NotImplementedError

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
        raise NotImplementedError
