from typing import List, Dict

from presidio_evaluator.evaluation import SpanError


class SpanEvaluationResult:
    def __init__(
        self,
        model_errors: List[SpanError],
        model_metrics: Dict[str, Dict],
    ):
        """
        """
        self.model_errors = model_errors,
        self.model_metrics = model_metrics