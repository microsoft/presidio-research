from dataclasses import dataclass
from typing import List, Dict

from presidio_evaluator.presidio_evaluator_2 import TokenOutput, SpanOutput


@dataclass
class SampleError:
    """
    Holds information about token and span errors for made a given sample for analysis purposes
    ...

    Attributes
    ----------
    full_text : str
        the full input text from InputSample
    metadata : Dict
        the metadata on text from InputSample
    token_output : List[TokenOutput]
        list of token errors of a given model for a sample
    span_output: List[SpanOutput]
        list of span outputs of a given model for a sample
    -------
    """

    def __init__(
            self,
            full_text: str,
            metadata: Dict = None,
            token_output: List[TokenOutput] = None,
            span_output: List[SpanOutput] = None
    ):
        """
        Constructs all the necessary attributes for the SampleError object
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
        """ Return str(self). """
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
