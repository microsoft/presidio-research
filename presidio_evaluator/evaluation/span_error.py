from presidio_evaluator import Span


class SpanError:
    def __init__(
        self,
        error_type: str,
        gold_span: Span,
        pred_span: Span,
        overlap_score: float,
        full_text: str
    ):
        """
        Holds information about model prediction output for analysis purposes
        """
        self.error_type = error_type
        self.gold_span = gold_span
        self.pred_span = pred_span
        self.overlap_score = overlap_score
        self.full_text = full_text