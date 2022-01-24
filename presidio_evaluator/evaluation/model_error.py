from typing import Dict, List

import pandas as pd
from spacy.tokens import Token


class ModelError:
    def __init__(
        self,
        error_type: str,
        annotation: str,
        prediction: str,
        token: Token,
        full_text: str,
        metadata: Dict,
    ):
        """
        Holds information about an error a model made for analysis purposes
        :param error_type: str, e.g. FP, FN, Person->Address etc.
        :param annotation: ground truth value
        :param prediction: predicted value
        :param token: token in question
        :param full_text: full input text
        :param metadata: metadata on text from InputSample
        """

        self.error_type = error_type
        self.annotation = annotation
        self.prediction = prediction
        self.token = token
        self.full_text = full_text
        self.metadata = metadata

    def __str__(self):
        return (
            "type: {}, "
            "Annotation = {}, "
            "prediction = {}, "
            "Token = {}, "
            "Full text = {}, "
            "Metadata = {}".format(
                self.error_type,
                self.annotation,
                self.prediction,
                self.token,
                self.full_text,
                self.metadata,
            )
        )

    def __repr__(self):
        return f"<ModelError {self.__str__()}"

    @staticmethod
    def most_common_fp_tokens(errors=List["ModelError"], n: int = 10, entity=None):
        """
        Print the n most common false positive tokens
        (tokens thought to be an entity)
        """
        fps = ModelError.get_false_positives(errors, entity)

        tokens = [err.token.text for err in fps]
        from collections import Counter

        by_frequency = Counter(tokens)
        most_common = by_frequency.most_common(n)
        print("Most common false positive tokens:")
        print(most_common)
        print("Example sentence with each FP token:")
        for tok, val in most_common:
            with_tok = [err for err in fps if err.token.text == tok]
            print(with_tok[0].full_text)

    @staticmethod
    def most_common_fn_tokens(errors=List["ModelError"], n: int = 10, entity=None):
        """
        Print all tokens that were missed by the model,
        including an example of the full text in which they appear.
        """
        fns = ModelError.get_false_negatives(errors, entity)

        fns_tokens = [err.token.text for err in fns]
        from collections import Counter

        by_frequency_fns = Counter(fns_tokens)
        most_common_fns = by_frequency_fns.most_common(n)
        print(most_common_fns)
        for tok, val in most_common_fns:
            with_tok = [err for err in fns if err.token.text == tok]
            print(
                "Token: {}, Annotation: {}, Full text: {}".format(
                    with_tok[0].token, with_tok[0].annotation, with_tok[0].full_text
                )
            )

    @staticmethod
    def get_errors_df(
        errors=List["ModelError"], entity: List[str] = None, error_type: str = "FN"
    ):
        """
        Get ModelErrors as pd.DataFrame
        """
        if error_type == "FN":
            filtered_errors = ModelError.get_false_negatives(errors, entity)
        elif error_type == "FP":
            filtered_errors = ModelError.get_false_positives(errors, entity)
        else:
            raise ValueError("error_type should be either FP or FN")

        if len(filtered_errors) == 0:
            print(
                "No errors of type {} and entity {} were found".format(
                    error_type, entity
                )
            )
            return None

        errors_df = pd.DataFrame.from_records(
            [error.__dict__ for error in filtered_errors]
        )
        metadata_df = pd.DataFrame(errors_df["metadata"].tolist())
        errors_df.drop(["metadata"], axis=1, inplace=True)
        new_errors_df = pd.concat([errors_df, metadata_df], axis=1)
        return new_errors_df

    @staticmethod
    def get_fps_dataframe(errors=List["ModelError"], entity: List[str] = None):
        """
        Get false positive ModelErrors as pd.DataFrame
        """
        return ModelError.get_errors_df(errors, entity, error_type="FP")

    @staticmethod
    def get_fns_dataframe(errors=List["ModelError"], entity: List[str] = None):
        """
        Get false negative ModelErrors as pd.DataFrame
        """
        return ModelError.get_errors_df(errors, entity, error_type="FN")

    @staticmethod
    def get_false_positives(errors=List["ModelError"], entity=None):
        """
        Get a list of all false positive errors in the results
        """
        if isinstance(entity, str):
            entity = [entity]

        if entity:
            return [
                model_error
                for model_error in errors
                if model_error.error_type == "FP" and model_error.prediction in entity
            ]
        else:
            return [
                model_error for model_error in errors if model_error.error_type == "FP"
            ]

    @staticmethod
    def get_false_negatives(errors=List["ModelError"], entity=None):
        """
        Get a list of all false positive negative errors in the results (False negatives and wrong entity detection)
        """
        if isinstance(entity, str):
            entity = [entity]
        if entity:
            return [
                model_error
                for model_error in errors
                if model_error.error_type != "FP" and model_error.annotation in entity
            ]
        else:
            return [
                model_error for model_error in errors if model_error.error_type != "FP"
            ]
