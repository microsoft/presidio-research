from typing import Optional, List

from presidio_evaluator import Span


class SpanOutput:
    def __init__(
        self,
        output_type: str,
        overlap_score: float,
        annotated_span: Optional[Span] = None,
        predicted_span: Optional[Span] = None
    ):
        """
        Holds information about span prediction output for analysis purposes
        :param error_type: str, e.g. strict, exact, partial, incorrect, miss, spurious.
        :param overlap_score: float, overlapping ratio between annotated_span and predicted_span
        :param annotated_span: str, actual span which comes from the annotated file, e.g. Address
        :param predicted_span: str, predicted span of a given model
        """
        self.output_type = output_type
        self.overlap_score = overlap_score
        self.annotated_span = annotated_span
        self.predicted_span = predicted_span

    def __repr__(self):
        return (
            f"Output type: {self.output_type}\n"
            f"Overlap score: {self.overlap_score}\n"
            f"Annotated span: {self.annotated_span}\n"
            f"Predicted span: {self.predicted_span}\n"
        )

    @staticmethod
    def get_span_output_by_type(outputs=List["SpanOutput"], 
                                error_type=str,
                                n: Optional[int]=None, 
                                entity=None) -> List["SpanOutput"]:
        """
        Print the n most common tokens by error type
        :param outputs: List of span errors in SpanOutput format.
        :param error_type: str, span error type, e.g. strict, exact, partial, incorrect, miss, spurious
        :param n: int, top n most common output to filter. If n is None, all token errors of error_type are returned.
        :param entity: str, List of entities to filter, e.g. Person, Address. If entity is None, all entities are returned.
        """
        return List["SpanOutput"]
        