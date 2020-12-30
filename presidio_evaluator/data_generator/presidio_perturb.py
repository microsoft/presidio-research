from typing import List, Set, Dict

from presidio_analyzer import RecognizerResult

from presidio_evaluator.data_generator import FakeDataGenerator

import pandas as pd


class PresidioPerturb(FakeDataGenerator):
    def __init__(
        self,
        fake_pii_df: pd.DataFrame,
        lower_case_ratio: float = 0.0,
        ignore_types: Set[str] = None,
        entity_dict: Dict[str, str] = None,
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
         :param entity_dict: Dictionary with mapping of entity names between Presidio and the fake_pii_df.
         For example, {"EMAIL_ADDRESS": "EMAIL"}
        """

        self.fake_pii = self.prep_fake_pii(self.original_pii_df)
        self.entity_dict = entity_dict

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

        delta = 0
        text = original_text
        for resp in presidio_response:
            start = resp.start + delta
            end = resp.end + delta
            entity_text = original_text[resp.start : resp.end]
            entity_type = resp.entity_type
            if self.entity_dict:
                if entity_type in self.entity_dict:
                    entity_type = self.entity_dict[entity_type]

            text = f"{text[:start]}{{{entity_type}}}{text[end:]}"
            delta += len(entity_type) + 2 - len(entity_text)
        self.templates = [text]
        return [
            sample.full_text
            for sample in self.sample_examples(
                count=count, genders=genders, namesets=namesets
            )
        ]
