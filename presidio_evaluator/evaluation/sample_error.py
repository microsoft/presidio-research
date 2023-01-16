from typing import Dict, List

import pandas as pd
from typing import List, Optional, Dict
from spacy.tokens import Token

from presidio_evaluator.evaluation import TokenOutput, SpanOutput


class SampleError:
    def __init__(
        self,
        full_text: str,
        metadata: Dict = None,
        token_output: Optional[List[TokenOutput]] = None,
        span_output: Optional[List[SpanOutput]] = None
    ):
        """
        Holds information about token and span errors for made a given sample for analysis purposes
        :param full_text: full input text from InputSample
        :param metadata: metadata on text from InputSample
        :param token_output: list of token errors of a given model for a sample
        :param span_output: list of span outputs of a given model for a sample 
        """
        self.full_text = full_text
        self.metadata = metadata
        self.token_output = token_output
        self.span_output = span_output

    def __str__(self):
        return (
            "Full text = {}, "
            "Token errors = {}, "
            "Span outputs = {}, "
            "Metadata = {}".format(
                self.full_text,
                self.token_output,
                self.span_output,
                self.metadata
            )
        )

    def __repr__(self):
        return f"<ModelError {self.__str__()}"
