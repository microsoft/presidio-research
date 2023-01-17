import dataclasses
import json
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from faker import Faker
from faker.providers import BaseProvider
from faker.typing import SeedType
from tqdm import tqdm

from presidio_evaluator.data_generator import raw_data_dir
from presidio_evaluator.data_generator.faker_extensions import (
    FakerSpansResult as FakeSentenceResult,
    NationalityProvider,
    OrganizationProvider,
    UsDriverLicenseProvider,
    IpAddressProvider,
    AddressProviderNew,
    SpanGenerator,
    RecordsFaker,
    PhoneNumberProviderNew,
    AgeProvider,
    ReligionProvider,
    HospitalProvider
)
from presidio_evaluator.data_generator.faker_extensions.datasets import load_fake_person_df

presidio_templates_file_path = raw_data_dir / "templates.txt"
presidio_additional_entity_providers = [IpAddressProvider,
                                        NationalityProvider,
                                        OrganizationProvider,
                                        UsDriverLicenseProvider,
                                        AgeProvider,
                                        AddressProviderNew,
                                        PhoneNumberProviderNew,
                                        ReligionProvider,
                                        HospitalProvider]


class SentenceFaker:
    def __init__(
            self,
            custom_faker: Optional[Faker] = None,
            locale: Optional[List[str]] = None,
            lower_case_ratio: float = 0.05,
    ):
        """
        Leverages Faker to create fake PII entities into predefined templates of structure: a b c {{PII}} d e f,
        e.g. "My name is {{first_name}}."
        :param custom_faker: A Faker object provided by the user
        :param locale: A locale object to create our own Faker instance if a custom one was not provided.
        :param lower_case_ratio: Percentage of names that should start with lower case

        :example:

        >>>from presidio_evaluator.data_generator import SentenceFaker

        >>>template = "I just moved to {{city}} from {{country}}"
        >>>fake_sentence_result = SentenceFaker().parse(template)
        >>>print(fake_sentence_result.fake)
        I just moved to North Kim from Ukraine
        >>>print(fake_sentence_result.spans)
        [{"value": "Ukraine", "start": 31, "end": 38, "type": "country"}, {"value": "North Kim", "start": 16, "end": 25, "type": "city"}]
        """
        if custom_faker and locale:
            raise ValueError("If a custom faker is passed, it's expected to have its locales loaded")

        if custom_faker:
            self.faker = custom_faker
        else:
            generator = (
                SpanGenerator()
            )  # To allow PresidioDataGenerator to return spans and not just strings
            self.faker = Faker(local=locale, generator=generator)
        self.lower_case_ratio = lower_case_ratio

    def parse(
            self, template: str, template_id: Optional[int] = None, add_spans: Optional[bool] = True
    ) -> Union[FakeSentenceResult, str]:
        """
        This function replaces known PII {{tokens}} in a template sentence
        with a fake value for each token and returns a sentence with fake PII.

        Examples:
            1. "My name is {{first_name_female}} {{last_name}}".
            2. "I want to increase limit on my card # {{credit_card_number}}
                for certain duration of time. is it possible?"


        :param template: str with token(s) that needs to be replaced by fake PII
        :param template_id: The identifier of the specific template
        :param add_spans: Whether to return the spans or just the fake text

        :returns: Fake sentence.

        """
        try:
            if isinstance(self.faker.factories[0], SpanGenerator):
                fake_pattern = self.faker.parse(
                    template, add_spans=add_spans, template_id=template_id
                )
            else:
                fake_pattern = self.faker.parse(template)
            if random.random() < self.lower_case_ratio:
                fake_pattern = self._lower_pattern(fake_pattern)
            return fake_pattern
        except Exception as err:
            raise AttributeError(
                f'Failed to generate fake data based on template "{template}".'
                f"You might need to add a new Faker provider! "
                f"{err}"
            )

    @staticmethod
    def _lower_pattern(pattern: Union[str, FakeSentenceResult]):
        if isinstance(pattern, str):
            return pattern.lower()
        elif isinstance(pattern, FakeSentenceResult):
            pattern.fake = pattern.fake.lower()
            for span in pattern.spans:
                span.value = str(span.value).lower()
            return pattern

    def seed(self, seed_value=42):
        Faker.seed(seed_value)
        self.faker.seed_instance(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)

    def add_provider_alias(self, provider_name: str, new_name: str) -> None:
        """
        Adds a copy of a provider, with a different name
        :param provider_name: Name of original provider
        :param new_name: New name
        :example:
        >>>add_provider_alias(provider_name="name", new_name="person")
        >>>self.faker.person()
        """
        original = getattr(self.faker, provider_name)

        new_provider = BaseProvider(self.faker)
        setattr(new_provider, new_name, original)
        self.faker.add_provider(new_provider)


class PresidioSentenceFaker:
    """
    A high level interface for generating fake sentences with entity metadata.
    By default, this leverages all the existing templates and additional providers in this library.
    :param: locale: The faker locale to use e.g. 'en_US'
    :param lower_case_ratio: Percentage of names that should start with lower case
    :param: entity_providers: Defaults to presidio_additional_entity_providers, a provided argument overrides this
    :param: sentence_templates: Defaults to presidio_templates_file_path, a provided argument overrides this
    :param: random_seed: A seed to make results reproducible between runs
    """
    PROVIDER_ALIASES = dict(name='person', credit_card_number='credit_card', date_of_birth='birthday')

    faker_to_presidio_entity_type = dict(person="PERSON",
                                         ip_address="IP_ADDRESS",
                                         us_driver_license="US_DRIVER_LICENSE",
                                         organization="ORGANIZATION",
                                         name_female="PERSON",
                                         address="STREET_ADDRESS",
                                         country="GPE",
                                         state="GPE",
                                         credit_card_number="CREDIT_CARD",
                                         city="GPE",
                                         street_name="STREET_ADDRESS",
                                         building_number="STREET_ADDRESS",
                                         name="PERSON",
                                         iban="IBAN_CODE",
                                         last_name="PERSON",
                                         last_name_male="PERSON",
                                         last_name_female="PERSON",
                                         first_name="PERSON",
                                         first_name_male="PERSON",
                                         first_name_female="PERSON",
                                         phone_number="PHONE_NUMBER",
                                         url="DOMAIN_NAME",
                                         ssn="US_SSN",
                                         email="EMAIL_ADDRESS",
                                         date_time="DATE_TIME",
                                         date_of_birth="DATE_TIME",
                                         day_of_week="DATE_TIME",
                                         year="DATE_TIME",
                                         name_male="PERSON",
                                         prefix_male="TITLE",
                                         prefix_female="TITLE",
                                         prefix="TITLE",
                                         nationality="NRP",
                                         nation_woman="NRP",
                                         nation_man="NRP",
                                         nation_plural="NRP",
                                         first_name_nonbinary="PERSON",
                                         postcode="STREET_ADDRESS",
                                         secondary_address="STREET_ADDRESS",
                                         job="TITLE",
                                         zipcode="ZIP_CODE",
                                         state_abbr="GPE",
                                         age="AGE")

    def __init__(self,
                 locale: str,
                 lower_case_ratio: float,
                 entity_providers: Optional[List[BaseProvider]] = None,
                 sentence_templates: Optional[List[str]] = None,
                 random_seed: Optional[SeedType] = None):
        self._sentence_templates = sentence_templates
        if not self._sentence_templates:
            self._sentence_templates = [line.strip() for line in presidio_templates_file_path.read_text().splitlines()]
        if entity_providers is None:
            entity_providers = presidio_additional_entity_providers

        fake_person_df = load_fake_person_df()
        faker = RecordsFaker(records=fake_person_df, locale=locale)

        for entity_provider in entity_providers:
            faker.add_provider(entity_provider)

        self._sentence_faker = SentenceFaker(custom_faker=faker, lower_case_ratio=lower_case_ratio)
        self._sentence_faker.seed(random_seed)
        for provider, alias in self.PROVIDER_ALIASES.items():
            self._sentence_faker.add_provider_alias(provider_name=provider, new_name=alias)

        self.fake_sentence_results = None

    def generate_new_fake_sentences(self, num_samples: int) -> List[FakeSentenceResult]:
        self.fake_sentence_results = []
        # Map faker generated entity types to Presidio entity types
        for _ in tqdm(range(num_samples), desc="Sampling"):
            template_id = random.choice(range(len(self._sentence_templates)))
            template = self._sentence_templates[template_id]
            fake_sentence_result = self._sentence_faker.parse(template, template_id)
            for span in fake_sentence_result.spans:
                span.type = self.faker_to_presidio_entity_type[span.type]
            for key, value in self.faker_to_presidio_entity_type.items():
                fake_sentence_result.template = fake_sentence_result.template.replace("{{%s}}" % key, "{{%s}}" % value)
            self.fake_sentence_results.append(fake_sentence_result)
        return self.fake_sentence_results


if __name__ == "__main__":
    sentence_faker = PresidioSentenceFaker(locale="en_US", lower_case_ratio=0.05, random_seed=42)
    fake_sentence_results = sentence_faker.generate_new_fake_sentences(num_samples=10000)
    repo_root = Path(__file__).parent.parent.parent
    output_file = repo_root / "data/presidio_data_generator_data.json"
    to_json = [dataclasses.asdict(pattern) for pattern in fake_sentence_results]
    with open("{}".format(output_file), "w+", encoding="utf-8") as f:
        json.dump(to_json, f, ensure_ascii=False, indent=2)
