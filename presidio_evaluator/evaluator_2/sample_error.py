from dataclasses import dataclass
from typing import List, Dict

from presidio_evaluator.evaluator_2 import TokenOutput, SpanOutput


@dataclass
class SampleError:
    def __init__(
        self,
        full_text: str,
        metadata: Dict = None,
        token_output: List[TokenOutput] = None,
        span_output: List[SpanOutput] = None,
        entities_to_keep: List[str] = None,
    ):
        """
        Constructs all the necessary attributes for the SampleError object
        :param full_text: full input text from InputSample
        :param metadata: metadata on text from InputSample
        :param token_output: list of token errors of a given model for a sample
        :param span_output: list of span outputs of a given model for a sample
        :param entities_to_keep: list of entities to keep
        """
        self.full_text = full_text
        self.metadata = metadata
        self.token_output = token_output
        self.span_output = span_output
        self.entities_to_keep = entities_to_keep

    def __str__(self):
        """Return str(self)."""
        return (
            "Full text = {}, "
            "Token errors = {}, "
            "Span outputs = {}, "
            "Entities to keep = {}, "
            "Metadata = {}".format(
                self.full_text, self.token_output, self.span_output,
                self.entities_to_keep, self.metadata
            )
        )

    def __repr__(self):
        return f"<ModelError {self.__str__()}"
