import inspect
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from faker import Faker

from presidio_evaluator.data_generator import SentenceFaker, PresidioSentenceFaker
from presidio_evaluator.data_generator.faker_extensions import RecordGenerator
from presidio_evaluator.data_generator.faker_extensions import providers


def test_generator_correct_output():
    test_data_dir = Path(__file__).parent / "data"

    fake_name_data = pd.read_csv(test_data_dir / "FakeNameGenerator.com_100.csv")
    # Convert column names to lowercase to match patterns
    SentenceFaker.update_fake_name_generator_df(fake_name_data)
    records = fake_name_data.to_dict(orient="records")
    generator = RecordGenerator(records=records)
    faker = Faker(generator=generator)
    data_generator = SentenceFaker(custom_faker=faker, lower_case_ratio=0.0)

    sentence_templates = SentenceFaker.read_template_file(test_data_dir / "templates.txt")
    fake_sentences = data_generator.generate_fake_data(
        templates=sentence_templates, n_samples=100
    )

    for sample in fake_sentences:
        assert sample.fake
        assert sample.template in sentence_templates
        assert sample.template_id >= 0


def test_new_provider_no_alias_raises_attribute_error():
    data_generator = SentenceFaker(lower_case_ratio=0.0)
    with pytest.raises(AttributeError):
        data_generator.parse("My doctor is {{doc_name}}", 0)


def test_new_provider_with_alias():
    data_generator = SentenceFaker(lower_case_ratio=0.0)
    data_generator.add_provider_alias("name", "doc_name")
    start_of_sentence = "My doctor is "
    res = data_generator.parse(template=f"{start_of_sentence}{{{{doc_name}}}}", template_id=0)
    assert res
    assert len(res.fake) > len(start_of_sentence)
    assert start_of_sentence in res.fake


def _get_classes_from_module(module):
    return [member[1] for member in inspect.getmembers(module, inspect.isclass)
            if member[1].__module__ == module.__name__]


@pytest.mark.parametrize('num_records', (1500, 3000))
def test_record_generator(num_records: int):
    standard_faker = Faker()
    default_faker_providers = standard_faker.providers
    presidio_providers = _get_classes_from_module(providers)

    record_generator = PresidioSentenceFaker(locale='en', lower_case_ratio=0)
    assert len(record_generator._sentence_templates) > 0, 'Did not load default sentence templates'

    expected_providers = deepcopy(default_faker_providers)
    expected_providers.extend(presidio_providers)
    expected_providers.extend([standard_faker.__getattr__(key)
                               for key in PresidioSentenceFaker.PROVIDER_ALIASES.keys()])
    actual_providers = record_generator._data_generator.faker.providers
    num_aliases = len(PresidioSentenceFaker.PROVIDER_ALIASES)
    actual_num_providers = len(actual_providers)
    expected_aliases = set(getattr(standard_faker, provider_name)
                           for provider_name in PresidioSentenceFaker.PROVIDER_ALIASES.keys())
    assert actual_num_providers == len(expected_providers), \
        f'Expected {len(presidio_providers)} presidio providers to be used and {num_aliases} aliases. ' \
        f'Faker has been extended with {actual_num_providers - len(default_faker_providers)} providers/aliases. ' \
        f'Expected Providers: {[provider.__name__ for provider in presidio_providers]} ' \
        f'Expected Aliases: {expected_aliases} '

    fake_records = record_generator.generate_new_fake_sentences(num_records)
    assert len(fake_records) == num_records
    for record in fake_records:
        assert record.fake
        assert record.template
        assert record.template_id >= 0
