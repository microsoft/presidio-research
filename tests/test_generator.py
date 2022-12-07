from pathlib import Path

import pandas as pd
import pytest
from faker import Faker

from presidio_evaluator.data_generator import PresidioDataGenerator, PresidioFakeRecordGenerator
from presidio_evaluator.data_generator.faker_extensions import RecordGenerator


def test_generator_correct_output():
    test_data_dir = Path(__file__).parent / "data"

    fake_name_data = pd.read_csv(test_data_dir / "FakeNameGenerator.com_100.csv")
    # Convert column names to lowercase to match patterns
    PresidioDataGenerator.update_fake_name_generator_df(fake_name_data)
    records = fake_name_data.to_dict(orient="records")
    generator = RecordGenerator(records=records)
    faker = Faker(generator=generator)
    data_generator = PresidioDataGenerator(custom_faker=faker, lower_case_ratio=0.0)

    sentence_templates = PresidioDataGenerator.read_template_file(test_data_dir / "templates.txt")
    fake_sentences = data_generator.generate_fake_data(
        templates=sentence_templates, n_samples=100
    )

    for sample in fake_sentences:
        assert sample.fake
        assert sample.template in sentence_templates
        assert sample.template_id >= 0


def test_new_provider_no_alias_raises_attribute_error():
    data_generator = PresidioDataGenerator(lower_case_ratio=0.0)
    with pytest.raises(AttributeError):
        data_generator.parse("My doctor is {{doc_name}}", 0)


def test_new_provider_with_alias():
    data_generator = PresidioDataGenerator(lower_case_ratio=0.0)
    data_generator.add_provider_alias("name", "doc_name")
    start_of_sentence = "My doctor is "
    res = data_generator.parse(template=f"{start_of_sentence}{{{{doc_name}}}}", template_id=0)
    assert res
    assert len(res.fake) > len(start_of_sentence)
    assert start_of_sentence in res.fake


@pytest.mark.parametrize('num_records', range(5))
def test_record_generator(num_records: int):
    default_num_faker_providers = len(Faker().providers)
    record_generator = PresidioFakeRecordGenerator(locale='en', lower_case_ratio=0)

    assert len(record_generator._sentence_templates) > 0, 'Did not load default sentence templates'
    num_record_generator_providers = len(record_generator._data_generator.faker.providers)
    assert num_record_generator_providers > default_num_faker_providers, 'Did not add Presidio entity providers'

    fake_records = record_generator.generate_new_fake_records(num_records)
    assert len(fake_records) == num_records
    for record in fake_records:
        assert record.fake
        assert record.template
        assert record.template_id >= 0
