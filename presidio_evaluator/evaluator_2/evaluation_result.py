from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd

from presidio_evaluator.evaluator_2 import SampleError


class EvaluationResult:
    """
    Holds the output of token and span evaluation for a given dataset
    ...

    Attributes
    ----------
    sample_errors : List[SampleError]
        contain the token, span errors and input text for further inspection
    token_confusion_matrix : Optional[Counter] = None
        list of objects of type Counter with structure {(actual, predicted) : count}
    token_model_metrics : Optional[Dict[str, Counter]] = None
        metrics calculated based on token results for the reference dataset
    span_model_metrics: Optional[Dict[str, Counter]] = None
        metrics calculated based on token results for the reference dataset
    -------
    """

    def __init__(
            self,
            sample_errors: List[SampleError] = None,
            token_confusion_matrix: Counter = None,
            token_model_metrics: Dict[str, Counter] = None,
            span_model_metrics: Dict[str, Counter] = None
    ):
        """
        Constructs all the necessary attributes for the EvaluationResult object
        :param sample_errors: contain the token, span errors and input text for further inspection
        :param token_confusion_matrix: List of objects of type Counter
        with structure {(actual, predicted) : count}
        :param token_model_metrics: metrics calculated based on token results
        :param span_model_metrics: metrics calculated based on span results
        """

        self.sample_errors = sample_errors
        self.token_confusion_matrix = token_confusion_matrix
        self.token_model_metrics = token_model_metrics
        self.span_model_metrics = span_model_metrics

    def to_log(self) -> Dict:
        """
        Reformat the EvaluationResult to log the output
        """
        pass

    def to_confusion_matrix(self) -> Tuple[List[str], List[List[int]]]:
        """
        Convert the EvaluationResult to display confusion matrix to the end user
        """
        pass

    @staticmethod
    def span_fb_score(precision: float, recall: float, beta: int = 2) -> float:
        """
        Calculate the span F1 score
        :param precision: span precision
        :param recall: span recall
        :param beta: which metric to compute (1 for F1, 2 for F2, etc.)
        """
        return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
