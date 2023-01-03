import pandas as pd
from typing import Optional, List
from spacy.tokens import Token

from presidio_evaluator import Span, InputSample


class TokenOutput:
    def __init__(
        self,
        error_type: str,
        annotation: str,
        prediction: str,
        token: Token,
    ):
        """
        Holds information about an error a model made for analysis purposes
        :param error_type: str, e.g. FP, FN, Person->Address etc.
        :param annotation: ground truth value
        :param prediction: predicted value
        :param token: token in question
        """

        self.error_type = error_type
        self.annotation = annotation
        self.prediction = prediction
        self.token = token

    def __str__(self):
        return (
            "type: {}, "
            "Annotation = {}, "
            "prediction = {}, "
            "Token = {}, "
        )

    def __repr__(self):
        return f"<ModelError {self.__str__()}"

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

class ModelPrediction:
    def __init__(
        self,
        input_sample: InputSample,
        predicted_tags: Optional[List[str]],
        predicted_spans: Optional[List[Span]]
    ):
        """
        Holds information about model prediction in both span and token level
        :params
        """
        self.input_sample = input_sample
        self.predicted_tags = predicted_tags
        self.predicted_spans = predicted_spans
