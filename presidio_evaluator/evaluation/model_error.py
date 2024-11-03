from typing import Dict, List, Optional, Union
from pprint import pprint

import pandas as pd
from spacy.tokens import Token


class ModelError:
    def __init__(
        self,
        error_type: str,
        annotation: str,
        prediction: str,
        token: Union[Token, str],
        full_text: str,
        sample_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Holds information about an error a model made for analysis purposes
        :param error_type: str, e.g. FP, FN, Wrong entity
        :param annotation: ground truth value
        :param prediction: predicted value
        :param token: token in question
        :param full_text: full input text
        :param sample_id: Id of the sample this error belongs to
        :param metadata: metadata on text from InputSample
        """

        self.error_type = error_type
        self.annotation = annotation
        self.prediction = prediction
        self.token = token.text if isinstance(token, Token) else token
        self.full_text = full_text
        self.sample_id = sample_id
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
    def most_common_fp_tokens(
        errors=List["ModelError"], n: int = 10, entity: Optional[str] = None
    ):
        """
        Print the n most common false positive tokens
        (tokens thought to be an entity)
        """
        fps = ModelError.get_false_positives(errors, entity)

        tokens = [err.token for err in fps]
        from collections import Counter

        by_frequency = Counter(tokens)
        most_common = by_frequency.most_common(n)

        print("Most common false positive tokens:")
        pprint(most_common)
        print("---------------")
        print("Example sentence with each FP token:")
        for tok, val in most_common:
            with_tok = [err for err in fps if err.token == tok]
            print(
                f"\t- {with_tok[0].full_text} (`{with_tok[0].token}` "
                f"pred as {with_tok[0].prediction})"
            )

        return most_common

    @staticmethod
    def most_common_fn_tokens(
        errors=List["ModelError"], n: int = 10, entity: Optional[str] = None
    ):
        """
        Print all tokens that were missed by the model,
        including an example of the full text in which they appear.
        """
        fns = ModelError.get_false_negatives(errors, entity)

        fns_tokens = [err.token for err in fns]
        from collections import Counter

        by_frequency_fns = Counter(fns_tokens)
        most_common_fns = by_frequency_fns.most_common(n)
        print("Most common false negative tokens:")
        pprint(most_common_fns)
        print("---------------")
        print("Example sentence with each FN token:")
        for tok, val in most_common_fns:
            with_tok = [err for err in fns if err.token == tok]
            print(
                f"\t- {with_tok[0].full_text} (`{with_tok[0].token}` "
                f"annotated as {with_tok[0].annotation})"
            )

        return most_common_fns

    @staticmethod
    def get_errors_df(
        errors=List["ModelError"],
        entity: Optional[str] = None,
        error_type: str = "FN",
        verbose: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get ModelErrors as pd.DataFrame
        :param errors: A list of ModelErrors
        :param entity: Entity to filter on
        :param error_type: Should be either FP, FN or `Wrong entity`
        :param verbose: True if should print
        """
        if error_type == "FN":
            filtered_errors = ModelError.get_false_negatives(errors, entity)
        elif error_type == "FP":
            filtered_errors = ModelError.get_false_positives(errors, entity)
        elif error_type == "Wrong entity":
            filtered_errors = ModelError.get_wrong_entities(errors, entity)
        else:
            raise ValueError("error_type should be either FP, FN or `Wrong entity`")

        if len(filtered_errors) == 0:
            if verbose:
                print(f"No errors of type {error_type} and entity {entity} were found")
            return None

        errors_df = pd.DataFrame.from_records(
            [error.__dict__ for error in filtered_errors]
        )
        metadata_df = pd.DataFrame(errors_df["metadata"].tolist())
        errors_df.drop(["metadata"], axis=1, inplace=True)
        new_errors_df = pd.concat([errors_df, metadata_df], axis=1)
        return new_errors_df

    @staticmethod
    def get_fps_dataframe(
        errors=List["ModelError"], entity: Optional[str] = None, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Get false positive ModelErrors as pd.DataFrame
        """
        return ModelError.get_errors_df(
            errors, entity, error_type="FP", verbose=verbose
        )

    @staticmethod
    def get_fns_dataframe(
        errors=List["ModelError"], entity: Optional[str] = None, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Get false negative ModelErrors as pd.DataFrame
        """
        return ModelError.get_errors_df(
            errors, entity, error_type="FN", verbose=verbose
        )

    @staticmethod
    def get_wrong_entity_dataframe(
        errors=List["ModelError"], entity: Optional[str] = None, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Get false negative ModelErrors as pd.DataFrame
        """
        return ModelError.get_errors_df(
            errors, entity, error_type="Wrong entity", verbose=verbose
        )

    @staticmethod
    def get_false_positives(
        errors=List["ModelError"], entity: Optional[str] = None
    ) -> List["ModelError"]:
        """
        Get a list of all false positive errors in the results
        """

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
    def get_false_negatives(
        errors=List["ModelError"], entity: Optional[str] = None
    ) -> List["ModelError"]:
        """
        Get a list of all false negative errors in the results
        """

        if entity:
            return [
                model_error
                for model_error in errors
                if model_error.error_type == "FN" and model_error.annotation in entity
            ]
        else:
            return [
                model_error for model_error in errors if model_error.error_type == "FN"
            ]

    @staticmethod
    def get_wrong_entities(
        errors=List["ModelError"], entity: Optional[str] = None
    ) -> List["ModelError"]:
        """
        Get a list of all mismatches in the results
        (wrong entity detection)
        """
        if entity:
            return [
                model_error
                for model_error in errors
                if model_error.error_type == "Wrong entity"
                and (
                    model_error.annotation in entity or model_error.prediction in entity
                )
            ]
        else:
            return [
                model_error
                for model_error in errors
                if model_error.error_type == "Wrong entity"
            ]
