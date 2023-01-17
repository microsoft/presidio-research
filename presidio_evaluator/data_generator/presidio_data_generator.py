import dataclasses
import json
import random
import re
from pathlib import Path
from typing import List, Optional, Union, Generator

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider
from faker.typing import SeedType
from pandas import DataFrame
from tqdm import tqdm

from presidio_evaluator.data_generator import raw_data_dir
from presidio_evaluator.data_generator.faker_extensions import (
    FakerSpansResult,
    NationalityProvider,
    OrganizationProvider,
    UsDriverLicenseProvider,
    IpAddressProvider,
    AddressProviderNew,
    SpanGenerator,
    RecordsFaker,
    PhoneNumberProviderNew,
    AgeProvider,
    ReligionProvider
)

presidio_templates_file_path = raw_data_dir / "templates.txt"
presidio_additional_entity_providers = [IpAddressProvider,
                                        NationalityProvider,
                                        OrganizationProvider,
                                        UsDriverLicenseProvider,
                                        AgeProvider,
                                        AddressProviderNew,
                                        PhoneNumberProviderNew,
                                        ReligionProvider]


class PresidioDataGenerator:
    def __init__(
            self,
            custom_faker: Optional[Faker] = None,
            locale: Optional[List[str]] = None,
            lower_case_ratio: float = 0.05,
    ):
        """
        Fake data generator.
        Leverages Faker to create fake PII entities into predefined templates of structure: a b c {{PII}} d e f,
        e.g. "My name is {{first_name}}."
        :param custom_faker: A Faker object provided by the user
        :param locale: A locale object to create our own Faker instance if a custom one was not provided.
        :param lower_case_ratio: Percentage of names that should start with lower case

        :example:

        >>>from presidio_evaluator.data_generator import PresidioDataGenerator

        >>>sentence_templates = [
        >>>    "My name is {{name}}",
        >>>    "Please send it to {{address}}",
        >>>    "I just moved to {{city}} from {{country}}"
        >>>]


        >>>data_generator = PresidioDataGenerator()
        >>>fake_records = data_generator.generate_fake_data(
        >>>    templates=sentence_templates, n_samples=10
        >>>)

        >>>fake_records = list(fake_records)

        >>># Print the spans of the first sample
        >>>print(fake_records[0].fake)
        I just moved to North Kim from Ukraine

        >>>print(fake_records[0].spans)
        [{"value": "Ukraine", "start": 31, "end": 38, "type": "country"}, {"value": "North Kim", "start": 16, "end": 25, "type": "city"}]

        """
        if custom_faker and locale:
            raise ValueError(
                "If a custom faker is passed, it's expected to have its locales loaded"
            )

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
    ) -> Union[FakerSpansResult, str]:
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
    def read_template_file(templates_file):
        with open(templates_file) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            return lines

    def generate_fake_data(
            self, templates: List[str], n_samples: int
    ) -> Union[Generator[FakerSpansResult, None, None], Generator[str, None, None]]:
        """
        Generates fake PII data whenever it encounters known faker entities in a template.
        :param templates: A list of strings containing templates
        :param n_samples: Number of samples to generate
        """
        if not templates:
            templates = None

        for _ in tqdm(range(n_samples), desc="Sampling"):
            template_id = random.choice(range(len(templates)))
            template = templates[template_id]
            yield self.parse(template, template_id)

    @staticmethod
    def _lower_pattern(pattern: Union[str, FakerSpansResult]):
        if isinstance(pattern, str):
            return pattern.lower()
        elif isinstance(pattern, FakerSpansResult):
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

    @staticmethod
    def update_fake_name_generator_df(fake_data: pd.DataFrame) -> DataFrame:
        """
        Adapts the csv from FakeNameGenerator to fit the data generation process used here.
        :param fake_data: a pd.DataFrame with loaded data from FakeNameGenerator.com
        :return: None
        """

        def camel_to_snake(name):
            # Borrowed from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
            name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        def full_name(row):
            if random.random() > 0.2:
                return f'{row.first_name} {row.last_name}'
            else:
                space_after_initials = " " if random.random() > 0.5 else ". "
                return f'{row.first_name} {row.middle_initial}{space_after_initials}{row.last_name}'

        def name_gendered(row):
            first_name_female, prefix_female, last_name_female = (
                (row.first_name, row.prefix, row.last_name)
                if row.gender == "female"
                else ("", "", "")
            )
            first_name_male, prefix_male, last_name_male = (
                (row.first_name, row.prefix, row.last_name)
                if row.gender == "male"
                else ("", "", "")
            )
            return (
                first_name_female,
                first_name_male,
                prefix_female,
                prefix_male,
                last_name_female,
                last_name_male,
            )

        fake_data.columns = [camel_to_snake(col) for col in fake_data.columns]

        # Update some column names to fit Faker
        fake_data.rename(
            columns={"country": "country_code", "state": "state_abbr"}, inplace=True
        )

        fake_data.rename(
            columns={
                "country_full": "country",
                "name_set": "nationality",
                "street_address": "street_name",
                "state_full": "state",
                "given_name": "first_name",
                "surname": "last_name",
                "title": "prefix",
                "email_address": "email",
                "telephone_number": "phone_number",
                "telephone_country_code": "country_calling_code",
                "birthday": "date_of_birth",
                "cc_number": "credit_card_number",
                "cc_type": "credit_card_provider",
                "cc_expires": "credit_card_expire",
                "occupation": "job",
                "domain": "domain_name",
                "username": "user_name",
                "zip_code": "zipcode",
            },
            inplace=True,
        )
        fake_data["person"] = fake_data.apply(full_name, axis=1)
        fake_data["name"] = fake_data["person"]
        genderized = fake_data.apply(
            lambda x: pd.Series(
                name_gendered(x),
                index=[
                    "first_name_female",
                    "first_name_male",
                    "prefix_female",
                    "prefix_male",
                    "last_name_female",
                    "last_name_male",
                ],
            ),
            axis=1,
            result_type="expand",
        )

        # Remove credit card data, rely on Faker's as it is more realistic
        del fake_data["credit_card_number"]

        fake_data = pd.concat([fake_data, genderized], axis="columns")
        return fake_data


class PresidioFakeRecordGenerator:
    """
    Fake record generator.
    Leverages PresidioDataGenerator and the existing templates and new providers in this library to give a high level
    interface for generating a list of fake records.
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
            self._sentence_templates = PresidioDataGenerator.read_template_file(presidio_templates_file_path)
        if entity_providers is None:
            entity_providers = presidio_additional_entity_providers

        fake_person_data_path = raw_data_dir / "FakeNameGenerator.com_3000.csv"
        fake_person_df = pd.read_csv(fake_person_data_path)
        fake_person_df = PresidioDataGenerator.update_fake_name_generator_df(fake_person_df)
        faker = RecordsFaker(records=fake_person_df, locale=locale)

        for entity_provider in entity_providers:
            faker.add_provider(entity_provider)

        self._data_generator = PresidioDataGenerator(custom_faker=faker, lower_case_ratio=lower_case_ratio)
        self._data_generator.seed(random_seed)
        for provider, alias in self.PROVIDER_ALIASES.items():
            self._data_generator.add_provider_alias(provider_name=provider, new_name=alias)

        self.fake_records = None

    def generate_new_fake_records(self, num_samples: int) -> List[FakerSpansResult]:
        self.fake_records = list(self._data_generator.generate_fake_data(templates=self._sentence_templates,
                                                                         n_samples=num_samples))
        # Map faker generated entity types to Presidio entity types
        for sample in self.fake_records:
            for span in sample.spans:
                span.type = self.faker_to_presidio_entity_type[span.type]
            for key, value in self.faker_to_presidio_entity_type.items():
                sample.template = sample.template.replace("{{%s}}" % key, "{{%s}}" % value)
        return self.fake_records


if __name__ == "__main__":
    entity_generator = PresidioFakeRecordGenerator(locale="en_US", lower_case_ratio=0.05,
                                                   random_seed=42)
    fake_patterns = entity_generator.generate_new_fake_records(num_samples=10000)
    repo_root = Path(__file__).parent.parent.parent
    output_file = repo_root / "data/presidio_data_generator_data.json"
    to_json = [dataclasses.asdict(pattern) for pattern in fake_patterns]
    with open("{}".format(output_file), "w+", encoding="utf-8") as f:
        json.dump(to_json, f, ensure_ascii=False, indent=2)
