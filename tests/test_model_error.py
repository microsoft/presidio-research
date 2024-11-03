import pytest

from presidio_evaluator.evaluation import ModelError


@pytest.fixture(scope="module")
def model_errors():
    model_errors = [ModelError(error_type="FP",
                               annotation="O",
                               prediction="LOC",
                               token="Bob",
                               full_text="Bob likes Jane"),
                   ModelError(error_type="FN",
                              annotation="LOC",
                              prediction="O",
                              token="Bob",
                              full_text="Bob likes Jane"),
                   ModelError(error_type="Wrong entity",
                              annotation="PER",
                              prediction="LOC",
                              token="Bob",
                              full_text="Bob likes Jane")]

    return model_errors



def test_model_error_most_common_fp_tokens(model_errors):
    most_common = ModelError.most_common_fp_tokens(model_errors)
    assert most_common[0] == ("Bob", 1)

def test_model_error_most_common_fn_tokens(model_errors):
    most_common = ModelError.most_common_fn_tokens(model_errors)
    assert most_common[0] == ("Bob", 1)


def test_get_false_positives_existing_entity_returns_errors(model_errors):
    fps = ModelError.get_false_positives(model_errors, "LOC")
    assert fps == [model_errors[0]]

def test_get_false_positives_no_entity_returns_none(model_errors):
    fps = ModelError.get_false_positives(model_errors, "XYZ")
    assert len(fps) == 0

def test_get_false_negatives_existing_entity_returns_errors(model_errors):
    fns = ModelError.get_false_negatives(model_errors, "LOC")
    assert fns == [model_errors[1]]

def test_get_false_negatives_no_entity_returns_none(model_errors):
    fns = ModelError.get_false_negatives(model_errors, "XYZ")
    assert len(fns) == 0

def test_get_wrong_entity_existing_entity_returns_errors(model_errors):
    wrong1 = ModelError.get_wrong_entities(model_errors, "LOC")
    assert wrong1 == [model_errors[2]]

    wrong2 = ModelError.get_wrong_entities(model_errors, "PER")
    assert wrong2 == [model_errors[2]]

def test_get_wrong_negatives_no_entity_returns_none(model_errors):
    wrong = ModelError.get_wrong_entities(model_errors, "XYZ")
    assert len(wrong) == 0


def test_get_errors_df(model_errors):
    fp_df = ModelError.get_errors_df(model_errors,entity="LOC", error_type="FP")
    assert fp_df["error_type"][0] == "FP"
    assert fp_df["annotation"][0] == "O"
    assert fp_df["prediction"][0] == "LOC"
    assert fp_df["token"][0] == "Bob"
    assert fp_df["full_text"][0] == "Bob likes Jane"

    fn_df = ModelError.get_errors_df(model_errors,entity="LOC", error_type="FN")
    assert fn_df["error_type"][0] == "FN"
    assert fn_df["annotation"][0] == "LOC"
    assert fn_df["prediction"][0] == "O"
    assert fn_df["token"][0] == "Bob"
    assert fn_df["full_text"][0] == "Bob likes Jane"

    wrong_df = ModelError.get_errors_df(model_errors,entity="LOC", error_type="Wrong entity")
    assert wrong_df["error_type"][0] == "Wrong entity"
    assert wrong_df["annotation"][0] == "PER"
    assert wrong_df["prediction"][0] == "LOC"
    assert wrong_df["token"][0] == "Bob"
    assert wrong_df["full_text"][0] == "Bob likes Jane"
