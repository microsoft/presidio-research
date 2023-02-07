from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Dict

from presidio_evaluator import Span
from presidio_evaluator.presidio_evaluator_2 import (TokenOutput,
                                                     SpanOutput,
                                                     ModelPrediction,
                                                     EvaluationResult)


class Evaluator:
    """
    Evaluate a PII detection model or a Presidio analyzer / recognizer
    ...

    Attributes
    ----------
    entities_to_keep : List[SampleError]
        contain the token, span errors and input text for further inspection
    compare_by_io : bool = True
        True if comparison should be done on the entity level and not the sub-entity level
    span_category_output : Dict
        a dictionary for storing the span metrics
    span_pii_eval: Dict[str, Counter]
        cover the four evaluation schemes for PII.
    span_entity_eval: Dict[str, Dict[str, Counter]]
        cover the four evaluation schemes for each entity in entities_to_keep.
    -------
    """

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

    def compare_token(self, model_prediction: ModelPrediction) -> Tuple[List[TokenOutput], Counter]:
        """
        Compares ground truth tags (annotation) and predicted (prediction) at token level.
        Return a list of TokenOutput and a list of objects of type Counter with structure {(actual, predicted) : count}
        :param model_prediction: model_prediction containing an InputSample and a list of predicted tags and tokens
        """
        raise NotImplementedError

    def compare_span(self, annotated_spans: List[Span], predicted_spans: List[Span]) -> Tuple[
                                    List[SpanOutput], Dict[str, Counter], Dict[str, Dict[str, Counter]]]:
        """
        Compares ground truth tags (annotation) and predicted (prediction) at span level.
        :param annotated_spans: truth annotation from InputSample
        :param predicted_spans: predicted span from PII model/system
        Returns:
        List[SpanOutput]: a list of SpanOutput
        dict: a dictionary of global PII results with structure {eval_type : {}}
        dict: a dictionary of PII results per entity with structure {entity_name: {eval_type : {}}}
        """
        raise NotImplementedError

    def evaluate_all(self, model_predictions: List[ModelPrediction]) -> EvaluationResult:
        """
        Evaluate the PII performance at token and span levels.
        :param model_predictions: list of ModelPrediction
        :returns:
        EvaluationResult: the evaluation outcomes in EvaluationResult format
        """
        raise NotImplementedError
