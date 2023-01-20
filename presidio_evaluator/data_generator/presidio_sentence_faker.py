import dataclasses
import json
import random
from pathlib import Path
from typing import List, Optional

from faker.providers import BaseProvider
from faker.typing import SeedType
from pandas import DataFrame
from tqdm import tqdm

from presidio_evaluator.data_generator import raw_data_dir
from presidio_evaluator.data_generator.faker_extensions import (
    FakerSpansResult as FakeSentenceResult,
    NationalityProvider,
    OrganizationProvider,
    UsDriverLicenseProvider,
    IpAddressProvider,
    AddressProviderNew,
    PhoneNumberProviderNew,
    AgeProvider,
    ReligionProvider,
    HospitalProvider
)
from presidio_evaluator.data_generator.faker_extensions.datasets import load_fake_person_df
from presidio_evaluator.data_generator.faker_extensions.sentences import SentenceFaker, RecordsFaker

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


class PresidioSentenceFaker:
    """
    A high level interface for generating fake sentences with entity metadata.
    By default, this leverages all the existing templates and additional providers in this library.
    :param: locale: The faker locale to use e.g. 'en_US'
    :param lower_case_ratio: Percentage of names that should start with lower case
    :param: sentence_templates: Defaults to presidio_templates_file_path, a provided argument overrides this
    :param: entity_providers: Defaults to presidio_additional_entity_providers, a provided argument overrides this
    :param: base_records: A DataFrame with entity types as columns and each row corresponding to a fake individual.
    Defaults to presidio_evaluator.data_generator.faker_extensions.datasets.load_fake_person_df()
    :param: random_seed: A seed to make results reproducible between runs
    """
    PROVIDER_ALIASES = dict(name='person', credit_card_number='credit_card', date_of_birth='birthday')
    ENTITY_TYPE_MAPPING = dict(person="PERSON",
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
                 sentence_templates: Optional[List[str]] = None,
                 entity_providers: Optional[List[BaseProvider]] = None,
                 base_records: Optional[DataFrame] = None,
                 random_seed: Optional[SeedType] = None):
        self._sentence_templates = sentence_templates
        if not self._sentence_templates:
            self._sentence_templates = [line.strip() for line in presidio_templates_file_path.read_text().splitlines()]
        if entity_providers is None:
            entity_providers = presidio_additional_entity_providers
        if base_records is None:
            base_records = load_fake_person_df()

        faker = RecordsFaker(records=base_records, locale=locale)
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
                span.type = self.ENTITY_TYPE_MAPPING[span.type]
            for key, value in self.ENTITY_TYPE_MAPPING.items():
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
