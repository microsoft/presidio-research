import sys

from typing import Optional, List
from collections import Counter

from presidio_evaluator import Span, InputSample


class TokenOutput:
    def __init__(
        self,
        error_type: str,
        annotated_tag: str,
        predicted_tag: str,
        token: str,
    ):
        """
        Holds information about a token error a model made for analysis purposes
        :param error_type: str, e.g. FP, FN, Person->Address etc.
        :param annotated_tag: str, actual label, e.g. Person
        :param predicted_tag: str, predicted label, e.g. Address
        :param token: str, token in question
        """

        self.error_type = error_type
        self.annotated_tag = annotated_tag
        self.predicted_tag = predicted_tag
        self.token = token

    def __str__(self):
        return (
            "type: {}, "
            "Annotated tag = {}, "
            "Predicted tag = {}, "
            "Token = {}".format(
                self.error_type,
                self.annotated_tag,
                self.predicted_tag,
                self.token
            )
        )

    def __repr__(self):
        return f"<TokenOutput {self.__str__()}"
    
    def __eq__(self, other):
        return (
            self.error_type == other.error_type
            and self.annotated_tag == other.annotated_tag
            and self.predicted_tag == other.predicted_tag
            and self.token == other.token
        )
    
    @staticmethod
    def get_common_token(errors=List["TokenOutput"], n: int = 10):
        """
        Print the n most common tokens errors that were missed by the model,
        including an example of full text in which they appear.
        """
        tokens = [err.token for err in errors]
        list_errors = []

        by_frequency = Counter(tokens)
        most_common = by_frequency.most_common(n)
        print(most_common)

        for tok, val in most_common:
            with_tok = [err for err in errors if err.token == tok]
            list_errors += with_tok
        
        return list_errors

    @staticmethod
    def get_token_error_by_type(errors=List["TokenOutput"], 
                                error_type=str,
                                entity=None,
                                n: Optional[int]=None) -> List["TokenOutput"]:
        """
        Print the n most common tokens by error type
        :param errors: List of token error in TokenOutput format.
        :param error_type: str, token error type, e.g. FP, FN
        :param n: int, top n most common error to filter. Default is None = all token errors of error_type are returned.
        :param entity: str, List of entities to filter, e.g. Person, Address. Default is None = all entities
        """
        if error_type not in ["FP", "FN"]:
            print("Invalid token error type...")
            sys.exit()
        # filter by error_type
        list_errors = [
                model_error
                for model_error in errors
                if model_error.error_type == error_type 
            ]
        # filter by entity list
        if entity:
            if error_type == "FP":
                list_errors = [
                    model_error
                    for model_error in list_errors
                    if model_error.predicted_tag in entity
                ]
            elif error_type == "FN":
                list_errors = [
                    model_error
                    for model_error in list_errors
                    if model_error.annotated_tag in entity
                ]
        # get top n of common error
        if n:
            list_errors = TokenOutput.get_common_token(errors=list_errors, n = n)
        return list_errors


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

    # TODO: Complete this function
    @staticmethod
    def get_span_output_by_type(outputs=List["SpanOutput"], 
                                output_type=str,
                                n: Optional[int]=None, 
                                entity=None) -> List["SpanOutput"]:
        """
        Print the n most common tokens by error type
        :param outputs: List of span output in SpanOutput format.
        :param output_type: str, span output type, e.g. STRICT, EXACT, ENT_TYPE, PARTIAL, SPURIOUS, MISS
        :param n: int, top n most common output to filter. Default is None = all token errors of error_type are returned.
        :param entity: str, List of entities to filter, e.g. Person, Address. Default is None = all entities.
        """
        if output_type not in ["STRICT", "EXACT", "ENT_TYPE", "PARTIAL", "SPURIOUS", "MISS"]:
            print("Invalid span output type...")
            sys.exit()
        # filter by output_type
        list_errors = [
                span_output
                for span_output in outputs
                if span_output.output_type == output_type 
            ]
        return list_errors
        

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