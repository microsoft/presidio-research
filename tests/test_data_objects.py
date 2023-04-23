import json
from pathlib import Path
from copy import deepcopy

import pytest
import spacy
from spacy.tokens import DocBin

from presidio_evaluator import InputSample, Span


@pytest.fixture(scope="session")
def small_dataset():
    dir_path = Path(__file__).parent
    input_samples = InputSample.read_dataset_json(
        Path(dir_path, "data", "generated_small.json")
    )
    return input_samples


@pytest.fixture(scope="session")
def input_sample_result():
    return InputSample(
        full_text="Dan is my name.",
        spans=[
            Span(
                entity_type="name", entity_value="Dan", start_position=0, end_position=3
            )
        ],
        masked="{{name}} is my name.",
        template_id=3,
        create_tags_from_span=True
    )


@pytest.fixture(scope="session")
def input_sample_result_2():
    return InputSample(
        full_text="Dan is my name. Tel Aviv is my home.",
        spans=[
            Span(
                entity_type="name", entity_value="Dan", start_position=0, end_position=3
            ),
            Span(
                entity_type="city",
                entity_value="Tel Aviv",
                start_position=16,
                end_position=24,
            ),
        ],
        masked="{{name}} is my name. {{city}} is my home.",
        template_id=4,
        create_tags_from_span=True
    )


def test_update_entity_types(input_sample_result):
    records = [deepcopy(input_sample_result)]
    [record.translate_input_sample_tags({"name": "PERSON"}) for record in records]
    assert records[0].spans[0].entity_type == "PERSON"


def test_load_dataset_from_file(input_sample_result_2):
    dir_path = Path(__file__).parent

    records = InputSample.read_dataset_json(Path(dir_path, "data", "mock_input_samples.json"))
    assert len(records) == 2
    assert records[0].masked == input_sample_result_2.masked
    assert records[0].full_text == input_sample_result_2.full_text
    assert records[0].spans == input_sample_result_2.spans


def test_count_entities(input_sample_result, input_sample_result_2):
    counts = InputSample.count_entities([input_sample_result, input_sample_result_2])
    assert len(counts) == 2
    assert all([ent[0] == "name" or ent[0] == "city" for ent in counts])


def test_remove_unsupported_entities(input_sample_result, input_sample_result_2):

    filtered = InputSample.remove_unsupported_entities(
        [input_sample_result, input_sample_result_2], {"name": "PERSON"}
    )
    assert len(filtered) == 1
    assert filtered[0].spans[0].entity_type == "name"


def test_to_conll():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        os.path.join(dir_path, "data/generated_small.json")
    )

    conll = InputSample.create_conll_dataset(input_samples)

    sentences = conll["sentence"].unique()
    assert len(sentences) == len(input_samples)


def test_to_spacy_all_entities(small_dataset):
    spacy_ver = InputSample.create_spacy_dataset(small_dataset)

    assert len(spacy_ver) == len(small_dataset)


def test_to_spacy_all_entities_specific_entities(small_dataset):
    spacy_ver = InputSample.create_spacy_dataset(small_dataset, entities=["PERSON"])

    spacy_ver_with_labels = [
        sample for sample in spacy_ver if len(sample[1]["entities"])
    ]

    assert len(spacy_ver_with_labels) < len(small_dataset)
    assert len(spacy_ver_with_labels) > 0


def test_to_spacy_file_and_back(small_dataset):
    spacy_pipeline = spacy.load("en_core_web_sm")
    InputSample.create_spacy_dataset(
        small_dataset,
        output_path="dataset.spacy",
        translate_tags=False,
        spacy_pipeline=spacy_pipeline,
        alignment_mode="strict",
    )

    db = DocBin()
    db.from_disk("dataset.spacy")
    docs = db.get_docs(vocab=spacy_pipeline.vocab)
    for doc, input_sample in zip(docs, small_dataset):
        input_ents = sorted(input_sample.spans, key=lambda x: x.start_position)
        spacy_ents = sorted(doc.ents, key=lambda x: x.start_char)
        for spacy_ent, input_span in zip(spacy_ents, input_ents):
            assert spacy_ent.start_char == input_span.start_position
            assert spacy_ent.end_char == input_span.end_position


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
