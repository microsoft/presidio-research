import pytest

from presidio_evaluator import InputSample
from presidio_evaluator.validation import split_by_template, get_samples_by_pattern, split_dataset


def get_mock_dataset():
    sample1 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 1})
    sample2 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 1})
    sample3 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 1})
    sample4 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 1})
    sample5 = InputSample("Bye there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 2})
    sample6 = InputSample("Bye there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 3})
    sample7 = InputSample("Bye there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 4})
    sample8 = InputSample("Bye there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 4})

    return [sample1, sample2, sample3, sample4, sample5, sample6, sample7, sample8]


def test_split_by_template():
    dataset = get_mock_dataset()
    train_templates, test_templates = split_by_template(dataset, 0.5)
    assert len(train_templates) == 2
    assert len(test_templates) == 2


def test_get_samples_by_pattern():
    dataset = get_mock_dataset()
    train_templates, test_templates = split_by_template(dataset, 0.5)
    train_samples = get_samples_by_pattern(dataset, train_templates)
    test_samples = get_samples_by_pattern(dataset, test_templates)

    dataset_templates = set([sample.metadata['Template#'] for sample in dataset])
    train_samples_templates = set([sample.metadata['Template#'] for sample in train_samples])
    test_samples_templates = set([sample.metadata['Template#'] for sample in test_samples])

    assert len(train_samples) + len(test_samples) == len(dataset)
    assert dataset_templates == train_samples_templates | test_samples_templates
    assert train_samples_templates & test_samples_templates == set()
    assert train_samples_templates == set(train_templates)
    assert test_samples_templates == set(test_templates)


def test_split_dataset_two_sets():
    sample1 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 1})
    sample2 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 2})
    sample3 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 3})
    sample4 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 4})
    train, test = split_dataset([sample1, sample2, sample3, sample4], [0.5, 0.5])
    assert len(train) == 2
    assert len(test) == 2


def test_split_dataset_four_sets():
    sample1 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 1})
    sample2 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 2})
    sample3 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 3})
    sample4 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 4})
    dataset = [sample1, sample2, sample3, sample4]
    train, test, val, dev = split_dataset(dataset, [0.25, 0.25, 0.25, 0.25])
    assert len(train) == 1
    assert len(test) == 1
    assert len(val) == 1
    assert len(dev) == 1


    # make sure all original template IDs are in the new sets

    original_keys = set([1, 2, 3, 4])
    t1 = set([sample.metadata['Template#'] for sample in train])
    t2 = set([sample.metadata['Template#'] for sample in test])
    t3 = set([sample.metadata['Template#'] for sample in dev])
    t4 = set([sample.metadata['Template#'] for sample in val])

    assert original_keys == t1 | t2 | t3 | t4


def test_split_dataset_test_with_0_ratio():
    sample1 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 1})
    sample2 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 2})
    sample3 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 3})
    sample4 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 4})
    dataset = [sample1, sample2, sample3, sample4]
    with pytest.raises(ValueError):
        train, test, zero = split_dataset(dataset, [0.5, 0.5, 0])


def test_split_dataset_test_with_smallish_ratio():
    sample1 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 1})
    sample2 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 2})
    sample3 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 3})
    sample4 = InputSample("Hi there", masked=None, spans=None, create_tags_from_span=False, metadata={"Template#": 4})
    dataset = [sample1, sample2, sample3, sample4]

    train, test, zero = split_dataset(dataset, [0.5, 0.4999995, 0.0000005])
    assert len(train) == 2
    assert len(test) == 2
    assert len(zero) == 0
