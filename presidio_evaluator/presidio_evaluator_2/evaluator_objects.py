from typing import Optional, List

from spacy.tokens import Token

from presidio_evaluator import Span, InputSample


class TokenOutput:
    """
    Holds information about a token error a model made for analysis purposes
    ...

    Attributes
    ----------
    error_type : str
        the type of error for a given token. e.g. FP, FN, Person->Address etc
    annotated_tag : str
        the actual label from annotated file
    predicted_tag : str
        the predicted label from PII system/model
    token: spacy Token
        the token in question
    Methods
    -------
    get_token_error_by_type(errors=List["TokenOutput"],
                            error_type=str,
                            entity: List[str] = None,
                            n: Optional[int] = None) -> List["TokenOutput"]:
        Print the n most common tokens by error type.
    """

    def __init__(
            self,
            error_type: str,
            annotated_tag: str,
            predicted_tag: str,
            token: Token,
    ):
        """
        Constructs all the necessary attributes for the TokenOutput object
        :param error_type: str, e.g. FP, FN, Person->Address etc.
        :param annotated_tag: str, actual label, e.g. Person
        :param predicted_tag: str, predicted label, e.g. Address
        :param token: spacy Token, token in question
        """

        self.error_type = error_type
        self.annotated_tag = annotated_tag
        self.predicted_tag = predicted_tag
        self.token = token

    def __str__(self) -> str:
        """ Return str(self). """
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

    def __repr__(self) -> str:
        """ Return repr(self). """
        return f"<TokenOutput {self.__str__()}"

    @staticmethod
    def get_token_error_by_type(errors=List["TokenOutput"],
                                error_type=str,
                                entity: List[str] = None,
                                n: Optional[int] = None) -> List["TokenOutput"]:
        """
        Print the n most common tokens by error type
        :param errors: List of token error in TokenOutput format.
        :param error_type: str, token error type, e.g. FP, FN
        :param n: int, top n most common error to filter. Default is None = all token errors of error_type are returned.
        :param entity: str, List of entities to filter, e.g. Person, Address. Default is None = all entities
        """
        pass


class SpanOutput:
    """
    Holds information about a span output a model made for analysis purposes
    ...

    Attributes
    ----------
    output_type : str
        the type of output type for a given Span. e.g. STRICT, EXACT, ENT_TYPE, PARTIAL, SPURIOUS, MISS.
    overlap_score : float
        overlapping ratio between annotated_span and predicted_span
    annotated_span : Optional[Span] = None
        actual span from the annotated data
    predicted_span: Optional[Span] = None
        predicted span from PII system/model
    Methods
    -------
    get_span_output_by_type(outputs=List["SpanOutput"],
                                error_type=str,
                                entity: List[str] = None) -> List["SpanOutput"]:
        Get the list of span output by output type.
    """

    def __init__(
            self,
            output_type: str,
            overlap_score: float,
            annotated_span: Span = None,
            predicted_span: Span = None
    ):
        """
        Constructs all the necessary attributes for the SpanOutput object
        :param output_type: str, e.g. STRICT, EXACT, ENT_TYPE, PARTIAL, SPURIOUS, MISS.
        :param overlap_score: float, overlapping ratio between annotated_span and predicted_span
        :param annotated_span: str, actual span which comes from the annotated file, e.g. Address
        :param predicted_span: str, predicted span of a given model
        """
        self.output_type = output_type
        self.overlap_score = overlap_score
        self.annotated_span = annotated_span
        self.predicted_span = predicted_span

    def __repr__(self) -> str:
        """ Return repr(self). """
        return (
            f"Output type: {self.output_type}\n"
            f"Overlap score: {self.overlap_score}\n"
            f"Annotated span: {self.annotated_span}\n"
            f"Predicted span: {self.predicted_span}\n"
        )

    @staticmethod
    def get_span_output_by_type(outputs=List["SpanOutput"],
                                output_type=str,
                                entity: List[str] = None) -> List["SpanOutput"]:
        """
        Get the list of span output by output type
        :param outputs: List of span errors in SpanOutput format.
        :param output_type: str, span error type, e.g. STRICT, EXACT, ENT_TYPE, PARTIAL, SPURIOUS, MISS.
        :param entity: List[str], List of entities to filter, e.g. ['Person', 'Address']. Default is None = all entities.
        """
        pass


class ModelPrediction:
    """
    Holds information of model prediction in both span and token level
    ...

    Attributes
    ----------
    input_sample : InputSample
        the type of error for a given token. e.g. FP, FN, Person->Address etc
    predicted_tags : Optional[List[str]]
        the actual label from annotated file
    predicted_spans : Optional[List[Span]]
        the predicted label from PII system/model
    Methods
    -------
    span_to_tag(predicted_spans: List[Span])-> List[str]:
        Turns a list of start and end values with corresponding labels, into a NER tagging (BILUO,BIO/IOB)
    tag_to_span(predicted_tags: List[str]) -> List[Span]:
        Turns a list of tokens with corresponding labels, into a list of span
    """

    def __init__(
            self,
            input_sample: InputSample,
            predicted_tags: Optional[List[str]],
            predicted_spans: Optional[List[Span]]
    ):
        """
        Constructs all the necessary attributes for the ModelPrediction object
        :params
        """
        self.input_sample = input_sample
        self.predicted_tags = predicted_tags
        self.predicted_spans = predicted_spans

    @staticmethod
    def span_to_tag(predicted_spans: List[Span]) -> List[str]:
        """
        Turns a list of start and end values with corresponding labels,
        into a list of NER tagging (BILUO,BIO/IOB)
        """
        pass

    @staticmethod
    def tag_to_span(predicted_tags: List[str]) -> List[Span]:
        """
        Turns a list of tokens with corresponding labels, into a list of span
        """
        pass
