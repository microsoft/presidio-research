from collections import Counter

import numpy as np
import pytest

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import TokenEvaluator, EvaluationResult
from tests.mocks import MockTokensModel, FiftyFiftyIdentityTokensMockModel, IdentityTokensMockModel


def test_evaluator_simple():
    prediction = ["O", "O", "O", "U-ANIMAL"]
    model = MockTokensModel(prediction=prediction, entities_to_keep=["ANIMAL"])

    evaluator = TokenEvaluator(model=model)
    sample = InputSample(
        full_text="I am the walrus", masked="I am the [ANIMAL]", spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus"]
    sample.tags = ["O", "O", "O", "U-ANIMAL"]

    evaluated = evaluator.evaluate_sample(sample, prediction)
    final_evaluation = evaluator.calculate_score([evaluated])

    assert final_evaluation.pii_precision == 1
    assert final_evaluation.pii_recall == 1


def test_evaluate_multiple_tokens_correct_statistics():
    prediction = ["O", "O", "O", "B-ANIMAL", "I-ANIMAL", "L-ANIMAL"]
    model = MockTokensModel(prediction=prediction)
    evaluator = TokenEvaluator(model=model, entities_to_keep=["ANIMAL"])
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
    evaluator = TokenEvaluator(model=model, entities_to_keep=["ANIMAL"])
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
    evaluator = TokenEvaluator(model=model, entities_to_keep=["ANIMAL"])
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
    evaluator = TokenEvaluator(model=model, entities_to_keep=["PERSON"], skip_words=["-"])
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

    evaluator = TokenEvaluator(model=model, entities_to_keep=["PERSON", "TENNIS_PLAYER"])
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
    evaluator = TokenEvaluator(model=model, entities_to_keep=["PERSON", "COMPANY"])
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
    evaluator = TokenEvaluator(model=model)
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
    evaluator = TokenEvaluator(model=model)
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
    evaluator = TokenEvaluator(model=model, entities_to_keep=["PERSON"])
    evaluation_results = evaluator.evaluate_all(input_samples)
    metrics = evaluator.calculate_score(evaluation_results)

    print(metrics.pii_precision)
    print(metrics.pii_recall)
    print(metrics.pii_f)

    assert metrics.pii_precision == 1
    assert metrics.pii_recall < 0.75
    assert metrics.pii_recall > 0.25


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
        (["!"], ["O"], ["U-PERSON"], np.nan, np.nan),
        ([], [], [], np.nan, np.nan),
    ],
)
def test_skip_words_are_not_counted_as_errors(
    tokens, tags, predicted_tags, precision, recall
):
    model = MockTokensModel(
        prediction=predicted_tags, entities_to_keep=["LOCATION", "PERSON"]
    )

    evaluator = TokenEvaluator(model=model)
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




def test_results_to_dataframe():
    prediction = ["O", "EMAIL", "PHONE", "LOCATION", "PERSON"]
    tokens = ["John", "details", "john@mail.com", "123-456-7890", "today"]
    tags = ["PERSON", "O", "EMAIL", "PHONE", "O"]
    start_indices = [0, 5, 13, 27, 40]
    evaluator = TokenEvaluator(model=MockTokensModel(prediction))

    sample = InputSample(
        full_text="John details john@mail.com 123-456-7890 today",
        tokens=tokens,
        start_indices=start_indices,
        tags=tags
    )

    results = evaluator.evaluate_all([sample, sample])

    df = evaluator.get_results_dataframe(results)
    expected_columns = ["sentence_id", "token", "annotation", "prediction", "start_indices"]
    for col in expected_columns:
        assert col in df.columns

    assert df["annotation"].to_list() == tags + tags
    assert df["prediction"].to_list() == prediction + prediction
    assert df["token"].to_list() == tokens + tokens
    assert df["sentence_id"].to_list() == [0]*len(tokens) + [1]*len(tokens)
    assert df["start_indices"].to_list() == start_indices + start_indices


def test_score_calculation():
    """
    Test that precision and recall calculations are correct:
    - FP and WrongEntity both hurt precision
    - Only FN hurts recall
    """
    prediction = ["PERSON", "PHONE", "O", "ORGANIZATION"]

    evaluator = TokenEvaluator(model=MockTokensModel(prediction))

    # Ground truth: [PERSON, O, EMAIL]
    # Prediction:   [PERSON, PHONE, LOCATION]
    sample = InputSample(
        full_text="John visited Paris France",
        tokens=["John", "visited", "Paris", "France"],
        tags=["PERSON", "O", "LOCATION", "LOCATION"],
    )

    result = evaluator.evaluate_sample(sample, prediction)
    score = evaluator.calculate_score([result])

    # Expected results:
    # TP: PERSON->PERSON
    # FP: O->PHONE
    # FN: Missing LOCATION
    # WrongEntity: LOCATION->ORGANIZATION

    # Wrong entities are handled differently for PII in general and individual entities

    # PII precision/recall don't take into account wrong entities (treat them as TP)
    # as we are interested in whether PII was detected or not, not the exact type.
    # Precision = (TP + WrongEntity) / (TP + WrongEntity + FP) = (1+1) / (1+1+1) = 0.667
    # Recall = (TP + WrongEntity) / (TP + WrongEntity + FN) = (1+1) / (1+1+1) = 0.667

    assert score.pii_precision == pytest.approx(0.66667 ,2)
    assert score.pii_recall == pytest.approx(0.66667, 2)

    # For individual entities, wrong entities are counted as FPs

    assert score.entity_precision_dict["PERSON"] == 1
    assert np.isnan(score.entity_precision_dict["LOCATION"])
    assert score.entity_precision_dict["PHONE"] == 0
    assert score.entity_precision_dict["ORGANIZATION"] == 0

    assert score.entity_recall_dict["PERSON"] == 1
    assert score.entity_recall_dict["LOCATION"] == 0
    assert np.isnan(score.entity_recall_dict["PHONE"])
    assert np.isnan(score.entity_recall_dict["ORGANIZATION"])


def test_calculate_score_existing_results_counter_individual_entities():
    results=Counter(
        {
            ("X", "X"): 50,
            ("Y", "Y"): 60,
            ("Z", "Z"): 70,
            ("X", "O"): 5,
            ("Y", "O"): 6,
            ("Z", "O"): 7,
            ("O", "X"): 5,
            ("O", "Y"): 6,
            ("O", "Z"): 7,
            ("X", "Y"): 5,
            ("X", "Z"): 5,
            ("Y", "X"): 6,
            ("Y", "Z"): 6,
            ("Z", "X"): 7,
            ("Z", "Y"): 7,
        }
    )
    x_tp = sum([results[i] for i in results if i[0] == "X" and i[1] == "X"])
    x_fp_tp = sum([results[i] for i in results if i[1] == "X"])
    x_fn_tp = sum([results[i] for i in results if i[0] == "X"])
    y_tp = sum([results[i] for i in results if i[0] == "Y" and i[1] == "Y"])
    y_fp_tp = sum([results[i] for i in results if i[1] == "Y"])
    y_fn_tp = sum([results[i] for i in results if i[0] == "Y"])
    z_tp = sum([results[i] for i in results if i[0] == "Z" and i[1] == "Z"])
    z_fp_tp = sum([results[i] for i in results if i[1] == "Z"])
    z_fn_tp = sum([results[i] for i in results if i[0] == "Z"])


    expected_x_precision=x_tp/x_fp_tp if x_fp_tp!=0 else np.nan
    expected_x_recall=x_tp/x_fn_tp if x_fn_tp!=0 else np.nan
    expected_y_precision=y_tp/y_fp_tp if y_fp_tp!=0 else np.nan
    expected_y_recall=y_tp/y_fn_tp if y_fn_tp!=0 else np.nan
    expected_z_precision=z_tp/z_fp_tp if z_fp_tp!=0 else np.nan
    expected_z_recall=z_tp/z_fn_tp if z_fn_tp!=0 else np.nan


    evaluator = TokenEvaluator(model=MockTokensModel(prediction=None))
    evaluation_score = evaluator.calculate_score(
        evaluation_results=[EvaluationResult(results)])

    assert evaluation_score.entity_precision_dict["X"] == expected_x_precision
    assert evaluation_score.entity_recall_dict["X"] == expected_x_recall
    assert evaluation_score.entity_precision_dict["Y"] == expected_y_precision
    assert evaluation_score.entity_recall_dict["Y"] == expected_y_recall
    assert evaluation_score.entity_precision_dict["Z"] == expected_z_precision
    assert evaluation_score.entity_recall_dict["Z"] == expected_z_recall
