from typing import Optional, List
from spacy.tokens import Token


class TokenOutput:
    def __init__(
        self,
        error_type: str,
        annotated_tag: str,
        predicted_tag: str,
        token: Token,
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

    @staticmethod
    def get_token_error_by_type(errors=List["TokenOutput"], 
                                error_type=str,
                                n: Optional[int]=None,
                                entity=None) -> List["TokenOutput"]:
        """
        Print the n most common tokens by error type
        :param errors: List of token error in TokenOutput format.
        :param error_type: str, token error type, e.g. FP, FN
        :param n: int, top n most common error to filter. If n is None, all token errors of error_type are returned.
        :param entity: str, List of entities to filter, e.g. Person, Address. If entity is None, all entities are returned.
        """
        return List["TokenOutput"]