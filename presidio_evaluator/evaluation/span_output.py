import pandas as pd
from typing import Optional, List

from presidio_evaluator import Span


class SpanOutput:
    def __init__(
        self,
        output_type: str,
        overlap_score: float,
        gold_span: Optional[Span] = None,
        pred_span: Optional[Span] = None
    ):
        """
        Holds information about span prediction output for analysis purposes
        :params 
        """
        self.output_type = output_type
        self.pred_span = pred_span
        self.gold_span = gold_span
        self.overlap_score = overlap_score

    def __repr__(self):
        return (
            f"Output type: {self.output_type}\n"
            f"Overlap score: {self.overlap_score}\n"
            f"Gold span: {self.gold_span}\n"
            f"Predicted span: {self.pred_span}\n"
        )
