from typing import List, Set, Dict

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from presidio_evaluator.data_generator import FakeDataGenerator

import pandas as pd


class PresidioPerturb(FakeDataGenerator):
    def __init__(
        self,
        fake_pii_df: pd.DataFrame,
        lower_case_ratio: float = 0.0,
        ignore_types: Set[str] = None,
    ):
        super().__init__(
            fake_pii_df=fake_pii_df,
            lower_case_ratio=lower_case_ratio,
            ignore_types=ignore_types,
            templates=None,
            span_to_tag=False,
        )
        """
        Gets a Presidio Analyzer response as input, and returns a list of sentences with fake PII entities
        :param fake_pii_df:
         A pd.DataFrame with a predefined set of PII entities as columns created using https://www.fakenamegenerator.com/
        :param lower_case_ratio: Percentage of names that should start
         with lower case
         :param ignore_types: set of types to ignore
        """

        self.fake_pii = self.prep_fake_pii(self.original_pii_df)

    def perturb(
        self,
        original_text: str,
        presidio_response: List[RecognizerResult],
        count: int,
        genders: List[str] = None,
        namesets: List[str] = None,
    ):
        """

        :param original_text: str containing the original text
        :param presidio_response: list of results from Presidio, to be used to know where entities are
        :param count: number of perturbations to return
        :param genders: gender valuse to use (options: 'female', 'male')
        :param namesets: name set values to use (options are values from the FakeNameGenerator NameSet column)
        :return: List[str] with fake perturbations of original text
        """

        presidio_response = sorted(presidio_response, key=lambda resp: resp.start)

        anonymizer_engine = AnonymizerEngine()
        anonymized_result = anonymizer_engine.anonymize(
            text=original_text, analyzer_results=presidio_response
        )

        text = anonymized_result.text
        text = text.replace(">", "}").replace("<", "{")

        self.templates = [text]
        return [
            sample.full_text
            for sample in self.sample_examples(
                count=count, genders=genders, namesets=namesets
            )
        ]
