import os
from pathlib import Path

import pandas as pd
import pytest
from faker import Faker

from presidio_evaluator.data_generator import PresidioDataGenerator
from presidio_evaluator.data_generator.faker_extensions import RecordGenerator


def test_generator_correct_output():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fake_pii_csv = Path(dir_path, "data/FakeNameGenerator.com_100.csv")
    template_file_path = Path(dir_path, "data/templates.txt")

    # Read FakeNameGenerator data
    fake_data = pd.read_csv(fake_pii_csv)
    # Convert column names to lowercase to match patterns
    PresidioDataGenerator.update_fake_name_generator_df(fake_data)
    records = fake_data.to_dict(orient="records")
    generator = RecordGenerator(records=records)

    # Create Faker and add additional specific providers
    faker = Faker(generator=generator)
    data_generator = PresidioDataGenerator(custom_faker=faker, lower_case_ratio=0.0)

    sentence_templates = PresidioDataGenerator.read_template_file(template_file_path)
    fake_sentences = data_generator.generate_fake_data(
        templates=sentence_templates, n_samples=100
    )

    for sample in fake_sentences:
        assert sample.fake
        assert sample.template
        assert sample.template_id >= 0


def test_new_provider_no_alias_raises_attribute_error():
    data_generator = PresidioDataGenerator(lower_case_ratio=0.0)

    with pytest.raises(AttributeError):
        data_generator.parse("My doctor is {{doc_name}}", 0)


def test_new_provider_with_alias():
    data_generator = PresidioDataGenerator(lower_case_ratio=0.0)
    data_generator.add_provider_alias("name", "doc_name")
    res = data_generator.parse(template="My doctor is {{doc_name}}", template_id=0)
    assert res
