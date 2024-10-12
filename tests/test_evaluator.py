from collections import Counter

import numpy as np
import pytest

from presidio_evaluator import InputSample, Span

from presidio_evaluator.evaluation import EvaluationResult, Evaluator
from tests.mocks import (
    IdentityTokensMockModel,
    FiftyFiftyIdentityTokensMockModel,
    MockTokensModel,
)


def test_evaluator_simple():
    prediction = ["O", "O", "O", "U-ANIMAL"]
    model = MockTokensModel(prediction=prediction, entities_to_keep=["ANIMAL"])

    evaluator = Evaluator(model=model)
    sample = InputSample(
        full_text="I am the walrus", masked="I am the [ANIMAL]", spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus"]
    sample.tags = ["O", "O", "O", "U-ANIMAL"]

    evaluated = evaluator.evaluate_sample(sample, prediction)
    final_evaluation = evaluator.calculate_score([evaluated])

    assert final_evaluation.pii_precision == 1
    assert final_evaluation.pii_recall == 1


def test_evaluate_sample_wrong_entities_to_keep_correct_statistics():
    prediction = ["O", "O", "O", "U-ANIMAL"]
    model = MockTokensModel(prediction=prediction)

    evaluator = Evaluator(model=model, entities_to_keep=["SPACESHIP"])

    sample = InputSample(
        full_text="I am the walrus", masked="I am the [ANIMAL]", spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus"]
    sample.tags = ["O", "O", "O", "U-ANIMAL"]

    evaluated = evaluator.evaluate_sample(sample, prediction)
    assert evaluated.results[("O", "O")] == 4


def test_evaluate_same_entity_correct_statistics():
    prediction = ["O", "U-ANIMAL", "O", "U-ANIMAL"]
    model = MockTokensModel(prediction=prediction)
    evaluator = Evaluator(model=model, entities_to_keep=["ANIMAL"], skip_words=["-"])
    sample = InputSample(
        full_text="I dog the walrus", masked="I [ANIMAL] the [ANIMAL]", spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus"]
    sample.tags = ["O", "O", "O", "U-ANIMAL"]

    evaluation_result = evaluator.evaluate_sample(sample, prediction)
    assert evaluation_result.results[("O", "O")] == 2
    assert evaluation_result.results[("ANIMAL", "ANIMAL")] == 1
    assert evaluation_result.results[("O", "ANIMAL")] == 1


def test_evaluate_multiple_entities_to_keep_correct_statistics():
    prediction = ["O", "U-ANIMAL", "O", "U-ANIMAL"]
    entities_to_keep = ["ANIMAL", "PLANT", "SPACESHIP"]
    model = MockTokensModel(prediction=prediction)
    evaluator = Evaluator(
        model=model, entities_to_keep=entities_to_keep, skip_words=["-"]
    )

    sample = InputSample(
        full_text="I dog the walrus", masked="I [ANIMAL] the [ANIMAL]", spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus"]
    sample.tags = ["O", "O", "O", "U-ANIMAL"]

    evaluation_result = evaluator.evaluate_sample(sample, prediction)
    assert evaluation_result.results[("O", "O")] == 2
    assert evaluation_result.results[("ANIMAL", "ANIMAL")] == 1
    assert evaluation_result.results[("O", "ANIMAL")] == 1


def test_evaluate_multiple_tokens_correct_statistics():
    prediction = ["O", "O", "O", "B-ANIMAL", "I-ANIMAL", "L-ANIMAL"]
    model = MockTokensModel(prediction=prediction)
    evaluator = Evaluator(model=model, entities_to_keep=["ANIMAL"])
    sample = InputSample(
        "I am the walrus amaericanus magnifico", masked=None, spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus", "americanus", "magnifico"]
    sample.tags = ["O", "O", "O", "B-ANIMAL", "I-ANIMAL", "L-ANIMAL"]

    evaluated = evaluator.evaluate_sample(sample, prediction)
    evaluation = evaluator.calculate_score([evaluated])

    assert evaluation.pii_precision == 1
    assert evaluation.pii_recall == 1


def test_evaluate_multiple_tokens_partial_match_correct_statistics():
    prediction = ["O", "O", "O", "B-ANIMAL", "L-ANIMAL", "O"]
    model = MockTokensModel(prediction=prediction)
    evaluator = Evaluator(model=model, entities_to_keep=["ANIMAL"])
    sample = InputSample(
        "I am the walrus amaericanus magnifico", masked=None, spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus", "americanus", "magnifico"]
    sample.tags = ["O", "O", "O", "B-ANIMAL", "I-ANIMAL", "L-ANIMAL"]

    evaluated = evaluator.evaluate_sample(sample, prediction)
    evaluation = evaluator.calculate_score([evaluated])

    assert evaluation.pii_precision == 1
    assert evaluation.pii_recall == 4 / 6


def test_evaluate_multiple_tokens_no_match_match_correct_statistics():
    prediction = ["O", "O", "O", "B-SPACESHIP", "L-SPACESHIP", "O"]
    model = MockTokensModel(prediction=prediction)
    evaluator = Evaluator(model=model, entities_to_keep=["ANIMAL"])
    sample = InputSample(
        "I am the walrus amaericanus magnifico", masked=None, spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus", "americanus", "magnifico"]
    sample.tags = ["O", "O", "O", "B-ANIMAL", "I-ANIMAL", "L-ANIMAL"]

    evaluated = evaluator.evaluate_sample(sample, prediction)
    evaluation = evaluator.calculate_score([evaluated])

    assert np.isnan(evaluation.pii_precision)
    assert evaluation.pii_recall == 0


def test_evaluate_multiple_examples_correct_statistics():
    prediction = ["U-PERSON", "O", "O", "U-PERSON", "O", "O"]
    model = MockTokensModel(prediction=prediction)
    evaluator = Evaluator(model=model, entities_to_keep=["PERSON"], skip_words=["-"])
    input_sample = InputSample("My name is Raphael or David", masked=None, spans=None)
    input_sample.tokens = ["My", "name", "is", "Raphael", "or", "David"]
    input_sample.tags = ["O", "O", "O", "U-PERSON", "O", "U-PERSON"]

    evaluated = evaluator.evaluate_all(
        [input_sample, input_sample, input_sample, input_sample]
    )
    scores = evaluator.calculate_score(evaluated)
    assert scores.pii_precision == 0.5
    assert scores.pii_recall == 0.5


def test_evaluate_multiple_examples_ignore_entity_correct_statistics():
    prediction = ["O", "O", "O", "U-PERSON", "O", "U-TENNIS_PLAYER"]
    model = MockTokensModel(prediction=prediction)

    evaluator = Evaluator(model=model, entities_to_keep=["PERSON", "TENNIS_PLAYER"])
    input_sample = InputSample("My name is Raphael or David", masked=None, spans=None)
    input_sample.tokens = ["My", "name", "is", "Raphael", "or", "David"]
    input_sample.tags = ["O", "O", "O", "U-PERSON", "O", "U-PERSON"]

    evaluated = evaluator.evaluate_all(
        [input_sample, input_sample, input_sample, input_sample]
    )
    scores = evaluator.calculate_score(evaluated)
    assert scores.pii_precision == 1
    assert scores.pii_recall == 1


def test_confusion_matrix_correct_metrics():
    from collections import Counter

    evaluated = [
        EvaluationResult(
            results=Counter(
                {
                    ("O", "O"): 150,
                    ("O", "PERSON"): 30,
                    ("O", "COMPANY"): 30,
                    ("PERSON", "PERSON"): 40,
                    ("COMPANY", "COMPANY"): 40,
                    ("PERSON", "COMPANY"): 10,
                    ("COMPANY", "PERSON"): 10,
                    ("PERSON", "O"): 30,
                    ("COMPANY", "O"): 30,
                }
            ),
            model_errors=None,
            text=None,
        )
    ]

    model = MockTokensModel(prediction=None)
    evaluator = Evaluator(model=model, entities_to_keep=["PERSON", "COMPANY"])
    scores = evaluator.calculate_score(evaluated, beta=2.5)

    assert scores.pii_precision == 0.625
    assert scores.pii_recall == 0.625
    assert scores.entity_recall_dict["PERSON"] == 0.5
    assert scores.entity_precision_dict["PERSON"] == 0.5
    assert scores.entity_recall_dict["COMPANY"] == 0.5
    assert scores.entity_precision_dict["COMPANY"] == 0.5


def test_confusion_matrix_2_correct_metrics():
    from collections import Counter

    evaluated = [
        EvaluationResult(
            results=Counter(
                {
                    ("O", "O"): 65467,
                    ("O", "ORG"): 4189,
                    ("GPE", "O"): 3370,
                    ("PERSON", "PERSON"): 2024,
                    ("GPE", "PERSON"): 1488,
                    ("GPE", "GPE"): 1033,
                    ("O", "GPE"): 964,
                    ("ORG", "ORG"): 914,
                    ("O", "PERSON"): 834,
                    ("GPE", "ORG"): 401,
                    ("PERSON", "ORG"): 35,
                    ("PERSON", "O"): 33,
                    ("ORG", "O"): 8,
                    ("PERSON", "GPE"): 5,
                    ("ORG", "PERSON"): 1,
                }
            ),
            model_errors=None,
            text=None,
        )
    ]

    model = MockTokensModel(prediction=None)
    evaluator = Evaluator(model=model)
    scores = evaluator.calculate_score(evaluated, beta=2.5)

    pii_tp = (
        evaluated[0].results[("PERSON", "PERSON")]
        + evaluated[0].results[("ORG", "ORG")]
        + evaluated[0].results[("GPE", "GPE")]
        + evaluated[0].results[("ORG", "GPE")]
        + evaluated[0].results[("ORG", "PERSON")]
        + evaluated[0].results[("GPE", "ORG")]
        + evaluated[0].results[("GPE", "PERSON")]
        + evaluated[0].results[("PERSON", "GPE")]
        + evaluated[0].results[("PERSON", "ORG")]
    )

    pii_fp = (
        evaluated[0].results[("O", "PERSON")]
        + evaluated[0].results[("O", "GPE")]
        + evaluated[0].results[("O", "ORG")]
    )

    pii_fn = (
        evaluated[0].results[("PERSON", "O")]
        + evaluated[0].results[("GPE", "O")]
        + evaluated[0].results[("ORG", "O")]
    )

    assert scores.pii_precision == pii_tp / (pii_tp + pii_fp)
    assert scores.pii_recall == pii_tp / (pii_tp + pii_fn)


def test_dataset_to_metric_identity_model():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        "{}/data/generated_small.json".format(dir_path), length=10
    )

    model = IdentityTokensMockModel()
    evaluator = Evaluator(model=model)
    evaluation_results = evaluator.evaluate_all(input_samples)
    metrics = evaluator.calculate_score(evaluation_results)

    assert metrics.pii_precision == 1
    assert metrics.pii_recall == 1


def test_dataset_to_metric_50_50_model():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        "{}/data/generated_small.json".format(dir_path), length=100
    )

    # Replace 50% of the predictions with a list of "O"
    model = FiftyFiftyIdentityTokensMockModel()
    evaluator = Evaluator(model=model, entities_to_keep=["PERSON"])
    evaluation_results = evaluator.evaluate_all(input_samples)
    metrics = evaluator.calculate_score(evaluation_results)

    print(metrics.pii_precision)
    print(metrics.pii_recall)
    print(metrics.pii_f)

    assert metrics.pii_precision == 1
    assert metrics.pii_recall < 0.75
    assert metrics.pii_recall > 0.25


def test_align_entity_types_correct_output():
    sample1 = InputSample(
        "I live in ABC",
        spans=[Span("A", "a", 0, 1), Span("A", "a", 10, 11), Span("B", "b", 100, 101)],
        create_tags_from_span=False,
    )
    sample2 = InputSample(
        "I live in ABC",
        spans=[Span("A", "a", 0, 1), Span("A", "a", 10, 11), Span("C", "c", 100, 101)],
        create_tags_from_span=False,
    )
    samples = [sample1, sample2]
    mapping = {
        "A": "1",
        "B": "2",
        "C": "1",
    }

    new_samples = Evaluator.align_entity_types(samples, mapping)

    count_per_entity = Counter()
    for sample in new_samples:
        for span in sample.spans:
            count_per_entity[span.entity_type] += 1

    assert count_per_entity["1"] == 5
    assert count_per_entity["2"] == 1


def test_align_entity_types_wrong_mapping_exception():
    sample1 = InputSample(
        "I live in ABC",
        spans=[Span("A", "a", 0, 1), Span("A", "a", 10, 11), Span("B", "b", 100, 101)],
        create_tags_from_span=False,
    )

    entities_mapping = {"Z": "z"}

    with pytest.raises(ValueError):
        Evaluator.align_entity_types(
            input_samples=[sample1], entities_mapping=entities_mapping
        )


@pytest.mark.parametrize(
    "tokens, tags, predicted_tags, precision, recall",
    [
        (
            ["John", "is", "in", "\n", "\t", "London"],
            ["U-PERSON", "O", "O", "B-LOCATION", "I-LOCATION", "I-LOCATION"],
            ["U-PERSON", "O", "O", "O", "O", "B-LOCATION"],
            1,
            1,
        ),
        (
            [">", ">>", ">>>", "Baku"],
            ["O", "O", "O", "U-LOCATION"],
            ["B-LOCATION", "I-LOCATION", "I-LOCATION", "L-LOCATION"],
            1,
            1,
        ),
        (
            ["Mr.", "", "Smith"],
            ["O", "O", "U-PERSON"],
            ["O", "B-PERSON", "I-PERSON"],
            1,
            1,
        ),
        (["!"], ["O"], ["U-PERSON"], np.NaN, np.NaN),
        ([], [], [], np.NaN, np.NaN),
    ],
)
def test_skip_words_are_not_counted_as_errors(
    tokens, tags, predicted_tags, precision, recall
):
    model = MockTokensModel(
        prediction=predicted_tags, entities_to_keep=["LOCATION", "PERSON"]
    )

    evaluator = Evaluator(model=model)
    sample = InputSample(full_text=" ".join(tokens), spans=None)
    sample.tokens = tokens
    sample.tags = tags

    evaluated = evaluator.evaluate_sample(sample, predicted_tags)
    final_evaluation = evaluator.calculate_score([evaluated])

    if np.isnan(precision):
        assert np.isnan(final_evaluation.pii_precision)
    else:
        assert final_evaluation.pii_precision == precision

    if np.isnan(recall):
        assert np.isnan(final_evaluation.pii_recall)
    else:
        assert final_evaluation.pii_recall == recall


@pytest.mark.parametrize(
    "tags, predicted_tags, expected_dict",
    [
        (
            ["O", "ID", "SSN"],
            ["O", "SSN", "SSN"],
            {("O", "O"): 1, ("SSN", "SSN"): 2},
        ),
        (
            ["O", "SSN", "SSN"],
            ["O", "ID", "SSN"],
            {("O", "O"): 1, ("SSN", "SSN"): 2},
        ),
        (
            ["O", "MID", "SSN"],
            ["O", "SSN", "SSN"],
            {("O", "O"): 1, ("MID", "SSN"): 1, ("SSN", "SSN"): 1},
        ),
    ],
)
def test_generic_entities_are_treated_like_specific_entities(
    tags, predicted_tags, expected_dict
):
    model = MockTokensModel(prediction=predicted_tags)
    evaluator = Evaluator(model=model)

    tokens = ["A", "123", "456"]

    sample = InputSample(full_text=" ".join(tokens), spans=None)
    sample.tokens = tokens
    sample.tags = tags

    evaluated = evaluator.evaluate_sample(sample, predicted_tags)

    assert evaluated.results == expected_dict
