from typing import Dict, List

import pandas as pd
from spacy.tokens import Token


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
