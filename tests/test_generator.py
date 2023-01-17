import inspect
from copy import deepcopy

import pytest
from faker import Faker

from presidio_evaluator.data_generator import SentenceFaker, PresidioSentenceFaker
from presidio_evaluator.data_generator.faker_extensions import providers


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


@pytest.mark.parametrize('num_sentences', (1500, 3000))
def test_generate_new_fake_sentences(num_sentences: int):
    standard_faker = Faker()
    default_faker_providers = standard_faker.providers
    presidio_providers = _get_classes_from_module(providers)

    sentence_faker = PresidioSentenceFaker(locale='en', lower_case_ratio=0)
    assert len(sentence_faker._sentence_templates) > 0, 'Did not load default sentence templates'

    expected_providers = deepcopy(default_faker_providers)
    expected_providers.extend(presidio_providers)
    expected_providers.extend([standard_faker.__getattr__(key)
                               for key in PresidioSentenceFaker.PROVIDER_ALIASES.keys()])
    actual_providers = sentence_faker._sentence_faker.faker.providers
    num_aliases = len(PresidioSentenceFaker.PROVIDER_ALIASES)
    actual_num_providers = len(actual_providers)
    expected_aliases = set(getattr(standard_faker, provider_name)
                           for provider_name in PresidioSentenceFaker.PROVIDER_ALIASES.keys())
    assert actual_num_providers == len(expected_providers), \
        f'Expected {len(presidio_providers)} presidio providers to be used and {num_aliases} aliases. ' \
        f'Faker has been extended with {actual_num_providers - len(default_faker_providers)} providers/aliases. ' \
        f'Expected Providers: {[provider.__name__ for provider in presidio_providers]} ' \
        f'Expected Aliases: {expected_aliases} '

    fake_sentence_results = sentence_faker.generate_new_fake_sentences(num_sentences)
    assert len(fake_sentence_results) == num_sentences
    for fake_sentence_result in fake_sentence_results:
        assert fake_sentence_result.fake
        assert fake_sentence_result.template
        assert fake_sentence_result.template_id >= 0
