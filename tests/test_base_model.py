import pytest

from presidio_evaluator import InputSample, Span
from tests.mocks import MockModel


@pytest.fixture(scope="session")
def mock_model():
    return MockModel(entity_mapping={"name": "new_name"}, entities_to_keep=["name"])


def test_align_entity_types(mock_model):
    input_sample = InputSample(
        full_text="Dan is my name.", spans=[Span("name", "Dan", 0, 3)]
    )

    mock_model.align_entity_types(sample=input_sample)

    assert input_sample.spans[0].entity_type == "new_name"


@pytest.mark.parametrize(
    "tags, expected_tags, ignore_unknown",
    [
        (["O", "O"], ["O", "O"], True),
        (["new_name", "O"], ["name", "O"], True),
        (["O", "credit_card"], ["O", "O"], True),
        (["O", "credit_card"], ["O", "credit_card"], False),
    ],
)
def test_align_prediction(mock_model, tags, expected_tags, ignore_unknown):
    actual_tags = mock_model.align_prediction_types(
        tags=tags, ignore_unknown=ignore_unknown
    )
    assert actual_tags == expected_tags


@pytest.mark.parametrize(
    "tags, expected_tags",
    [
        (["O", "O"], ["O", "O"]),
        (["name", "O"], ["name", "O"]),
        (["O", "credit_card"], ["O", "O"]),
    ],
)
def test_filter_tags_in_supported_entities(mock_model, tags, expected_tags):
    actual_tags = mock_model.filter_tags_in_supported_entities(tags=tags)
    assert actual_tags == expected_tags


@pytest.mark.parametrize(
    "tags, expected_tags, scheme",
    [
        (["O", "O"], ["O", "O"], "BILUO"),
        (["B-name", "I-name", "L-name"], ["B-name", "I-name", "I-name"], "BIO"),
        (["B-name", "I-name", "I-name"], ["B-name", "I-name", "L-name"], "BILUO"),
    ],
)
def test_to_scheme(mock_model, tags, expected_tags, scheme):
    mock_model.labeling_scheme = scheme
    actual_tags = mock_model.to_scheme(tags=tags)
    assert actual_tags == expected_tags


def test_to_log(mock_model):
    log_dict = mock_model.to_log()

    assert log_dict['labeling_scheme'] == mock_model.labeling_scheme
    assert log_dict['entities_to_keep'] == mock_model.entities
