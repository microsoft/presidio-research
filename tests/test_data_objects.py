import pytest
import spacy

from presidio_evaluator import InputSample, Span

from presidio_evaluator.data_generator.faker_extensions import (
    FakerSpansResult,
    FakerSpan,
)


@pytest.fixture(scope="session")
def faker_span_result():
    return FakerSpansResult(
        fake="Dan is my name.",
        spans=[FakerSpan("Dan", 0, 3, "name")],
        template="{{name}} is my name.",
        template_id=3,
    )


def test_to_conll():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        os.path.join(dir_path, "data/generated_small.json")
    )

    conll = InputSample.create_conll_dataset(input_samples)

    sentences = conll["sentence"].unique()
    assert len(sentences) == len(input_samples)


def test_to_spacy_all_entities():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        os.path.join(dir_path, "data/generated_small.json")
    )

    spacy_ver = InputSample.create_spacy_dataset(input_samples)

    assert len(spacy_ver) == len(input_samples)


def test_to_spacy_all_entities_specific_entities():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        os.path.join(dir_path, "data/generated_small.json")
    )

    spacy_ver = InputSample.create_spacy_dataset(input_samples, entities=["PERSON"])

    spacy_ver_with_labels = [
        sample for sample in spacy_ver if len(sample[1]["entities"])
    ]

    assert len(spacy_ver_with_labels) < len(input_samples)
    assert len(spacy_ver_with_labels) > 0


def test_to_spacy_json():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        os.path.join(dir_path, "data/generated_small.json")
    )

    spacy_ver = InputSample.create_spacy_json(input_samples)

    assert len(spacy_ver) == len(input_samples)
    assert "id" in spacy_ver[0]
    assert "paragraphs" in spacy_ver[0]


def test_faker_spans_result_to_input_sample(faker_span_result):

    input_sample = InputSample.from_faker_spans_result(
        faker_span_result, create_tags_from_span=False
    )

    assert input_sample.full_text == "Dan is my name."
    assert input_sample.masked == "{{name}} is my name."
    assert input_sample.spans[0] == Span("name", "Dan", 0, 3)
    assert input_sample.spans[0] == Span("name", "Dan", 0, 3)


def test_faker_spans_to_input_sample_with_tags(faker_span_result):
    input_sample = InputSample.from_faker_spans_result(
        faker_span_result, create_tags_from_span=True, scheme="BILUO"
    )
    assert input_sample.tags
    assert input_sample.tokens
    assert any(["I-name" in tag for tag in input_sample.tags])


def test_from_spacy_doc():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Nice to meet you Mr. Perkins.")

    sample = InputSample.from_spacy_doc(doc)
    assert sample.spans[0].entity_type == "PERSON"
    assert sample.tags == ["O", "O", "O", "O", "O", "U-PERSON", "O"]


@pytest.mark.parametrize(
    "start1, end1, start2, end2, intersection_length, ignore_entity_type",
    [
        (150, 153, 160, 165, 0, True),
        (150, 153, 150, 153, 3, True),
        (150, 153, 152, 154, 1, True),
        (150, 153, 100, 151, 1, True),
        (150, 153, 100, 151, 0, False),
    ],
)
def test_spans_intersection(
    start1, end1, start2, end2, intersection_length, ignore_entity_type
):
    span1 = Span(
        entity_type="A", entity_value="123", start_position=start1, end_position=end1
    )
    span2 = Span(
        entity_type="B", entity_value="123", start_position=start2, end_position=end2
    )

    intersection = span1.intersect(span2, ignore_entity_type=ignore_entity_type)
    assert intersection == intersection_length
