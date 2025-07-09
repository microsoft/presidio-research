import numpy as np
import pandas as pd
import pytest

from presidio_evaluator.evaluation import EvaluationResult, ErrorType, SpanEvaluator
from tests.mocks import MockModel


@pytest.fixture
def span_evaluator():
    """Create a SpanEvaluator instance for testing."""
    return SpanEvaluator(model=MockModel(), iou_threshold=0.75, char_based=True, skip_words=[])

# Test Scenario group 1: single span overlaps

def assert_error_types(expected_error_types, result, scenario):
    # Check error types
    if expected_error_types:
        error_types = [error.error_type for error in result.model_errors]
        for expected_type in expected_error_types:
            assert expected_type in error_types, (
                f"In {scenario}, expected error type {expected_type} not found in {error_types}"
            )


def assert_confusion_matrix(expected_results, result, scenario):
    # Check confusion matrix results
    for (ann, pred), expected_count in expected_results.items():
        actual_count = result.results.get((ann, pred), 0)
        assert actual_count == expected_count, (
            f"In {scenario}, confusion matrix entry ({ann}, {pred}) "
            f"expected to be {expected_count}, got {actual_count}"
        )


def assert_metric(expected_pii_metric, metric_name, result, scenario):

    metric = ""
    match metric_name:
        case "precision":
            metric = result.pii_precision
        case "recall":
            metric = result.pii_recall
        case "f":
            metric = result.pii_f
        case "_":
            raise ValueError(f"Unknown metric name: {metric_name}")

    if np.isnan(expected_pii_metric):
        assert np.isnan(metric), (
            f"In {scenario}, expected F1 score to be None, got {metric}"
        )
    else:
        assert metric == pytest.approx(expected_pii_metric, 3), (
            f"In {scenario}, expected {metric_name} score {expected_pii_metric}, got {metric}"
        )



@pytest.mark.parametrize(
    "scenario, annotation, prediction, tokens, start_indices, expected_tp, expected_fp, expected_fn, expected_results, expected_error_types",
    [
        # Scenario 1: Same type overlap with IoU > threshold
        (
                "Scenario 1: Same type overlap with IoU > threshold",
                ["O", "PERSON", "PERSON", "O"],
                ["O", "PERSON", "PERSON", "O"],
                ["The", "John", "Smith", "visited"],
                [0, 4, 9, 15],
                1,  # true positives
                0,  # false positives
                0,  # false negatives
                {("PERSON", "PERSON"): 1},  # confusion matrix
                [],  # no errors
        ),
        # Scenario 2: No overlap with annotated
        (
                "Scenario 2: No overlap with annotated",
                ["O", "PERSON", "PERSON", "O"],
                ["O", "O", "O", "O"],
                ["The", "John", "Smith", "visited"],
                [0, 4, 9, 15],
                0,  # true positives
                0,  # false positives
                1,  # false negatives
                {("PERSON", "O"): 1},  # confusion matrix
                [ErrorType.FN],  # error types
        ),
        # Scenario 3: No overlap with predicted
        (
                "Scenario 3: No overlap with predicted",
                ["O", "O", "O", "O"],
                ["O", "PERSON", "PERSON", "O"],
                ["The", "John", "Smith", "visited"],
                [0, 4, 9, 15],
                0,  # true positives
                1,  # false positives
                0,  # false negatives
                {("O", "PERSON"): 1},  # confusion matrix
                [ErrorType.FP],  # error types
        ),
        # Scenario 4: Different type overlap with IoU > threshold
        (
                "Scenario 4: Different type overlap with IoU > threshold",
                ["O", "LOCATION", "LOCATION", "O"],
                ["O", "PERSON", "PERSON", "O"],
                ["The", "New", "York", "visited"],
                [0, 4, 8, 13],
                0,  # true positives
                1,  # false positives
                1,  # false negatives
                {("LOCATION", "PERSON"): 1},  # confusion matrix
                [ErrorType.WrongEntity],  # error types
        ),
        # Scenario 5a: Same type overlap with IoU < threshold
        (
                "Scenario 5a: Same type overlap with IoU < threshold",
                ["O", "PERSON", "PERSON", "PERSON", "O"],
                ["O", "PERSON", "O", "O", "O"],
                ["The", "John", "Smith", "Johnson", "visited"],
                [0, 4, 9, 15, 23],
                0,  # true positives
                0,  # false positives
                1,  # false negatives
                {("PERSON", "O"): 1},  # confusion matrix
                [ErrorType.FN],  # error types
        ),
        # Scenario 5b: Different type overlap with IoU < threshold
        (
                "Scenario 5b: Different type overlap with IoU < threshold",
                ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"],
                ["O", "PERSON", "O", "O", "O"],
                ["The", "New", "York", "Mets", "visited"],
                [0, 4, 8, 13, 18],
                0,  # true positives
                1,  # false positives
                1,  # false negatives
                {("ORGANIZATION", "O"): 1, ("O", "PERSON"): 1},  # confusion matrix
                [ErrorType.FN, ErrorType.FP],  # error types
        ),
    ],
)
def test_scenario_group1(
        span_evaluator,
        scenario,
        annotation,
        prediction,
        tokens,
        start_indices,
        expected_tp,
        expected_fp,
        expected_fn,
        expected_results,
        expected_error_types,
):
    """Test each scenario in Group 1: single span overlaps."""
    # Build the DataFrame expected by SpanEvaluator
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "prediction": prediction,
            "start_indices": start_indices,
        }
    )

    # Run evaluation
    result = EvaluationResult()
    result = span_evaluator.calculate_score_on_df(
        per_type=True, results_df=df, evaluation_result=result
    )

    # Check true positives, false positives, and false negatives
    total_tp = sum(pii_type.true_positives for pii_type in result.per_type.values())
    total_fp = sum(pii_type.false_positives for pii_type in result.per_type.values())
    total_fn = sum(pii_type.false_negatives for pii_type in result.per_type.values())

    assert total_tp == expected_tp, f"In {scenario}, expected {expected_tp} TPs, got {total_tp}"
    assert total_fp == expected_fp, f"In {scenario}, expected {expected_fp} FPs, got {total_fp}"
    assert total_fn == expected_fn, f"In {scenario}, expected {expected_fn} FNs, got {total_fn}"

    # Check confusion matrix results
    for (ann, pred), expected_count in expected_results.items():
        actual_count = result.results.get((ann, pred), 0)
        assert actual_count == expected_count, (
            f"In {scenario}, confusion matrix entry ({ann}, {pred}) "
            f"expected to be {expected_count}, got {actual_count}"
        )

    # Check error types
    if expected_error_types:
        error_types = [error.error_type for error in result.model_errors]
        for expected_type in expected_error_types:
            assert expected_type in error_types, (
                f"In {scenario}, expected error type {expected_type} not found in {error_types}"
            )

        assert len(result.model_errors) == len(expected_error_types)


# Test Scenario group 2: annotated spans overlapping with multiple prediction spans

@pytest.mark.parametrize(
    "scenario, annotation, prediction, tokens, start_indices, expected_tp, expected_fp, expected_fn, expected_results, expected_error_types",
    [
        # Scenario 6A: Cumulative IoU with spans of the same type > threshold
        (
            "Scenario 6A: Cumulative IoU with spans of the same type > threshold",
            ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"],
            ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"],
            ["The", "New", "York", "Mets", "visited"],
            [0, 4, 8, 13, 18],
            1,  # true positives (combined spans make a good match)
            0,  # false positives
            0,  # false negatives
            {("ORGANIZATION", "ORGANIZATION"): 1},  # confusion matrix
            [],  # no errors
        ),
        # Scenario 6B: Cumulative IoU with spans of the same type < threshold
        (
            "Scenario 6B: Cumulative IoU with spans of the same type < threshold",
            ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"],
            ["O", "ORGANIZATION", "O", "ORGANIZATION", "O"],
            ["The", "New", "York", "Mets", "visited"],
            [0, 4, 8, 13, 18],
            0,  # true positives
            0,  # false positives (not counted as FP due to being same type)
            1,  # false negatives
            {("ORGANIZATION", "O"): 1,
             },  # confusion matrix
            [ErrorType.FN],  # errors
        ),
        # Scenario 7A: Cumulative IoU with spans of different types > threshold
        (
            "Scenario 7A: Cumulative IoU with spans of different types > threshold",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "PERSON", "LOCATION", "LOCATION", "O"],
            ["The", "John", "Smith", "Johnson", "visited"],
            [0, 4, 9, 17, 25],
            0,  # true positives
            1,  # false positives (for the wrong type)
            1,  # false negatives
            {("PERSON", "LOCATION"): 1},  # confusion matrix
            [ErrorType.WrongEntity],  # errors
        ),
        # Scenario 7B: Cumulative IoU with spans of different types < threshold
        (
            "Scenario 7B: Cumulative IoU with spans of different types < threshold",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "O", "LOCATION", "O", "PERSON"],
            ["The", "John", "Smith", "Johnson", "visited"],
            [0, 4, 9, 17, 25],
            0,  # true positives
            2,  # false positives
            1,  # false negatives
            {("PERSON", "O"): 1, ("O", "LOCATION"): 1, ("O", "PERSON"): 1},  # confusion matrix
            [ErrorType.FN, ErrorType.FP, ErrorType.FP],  # errors
        ),
        # Mixed case with both same and different types
        (
            "Mixed case: overlapping with both same and different entity types",
            ["O", "PERSON", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "PERSON", "O", "LOCATION", "PERSON", "O"],
            ["The", "John", "C", "B", "Johnson", "visited"],
            [0, 4, 9, 11, 13, 21],
            1,  # true positives (the parts that match)
            1,  # false positives (the wrong type)
            0,  # false negatives (since cumulative IoU is good)
            {("PERSON", "PERSON"): 1, ("O", "LOCATION"): 1},  # confusion matrix. (O, LOCATION) because of low IoU
            [ErrorType.FP],  # Not WrongEntity since cumulative IoU between Location and Person is low
        ),
    ],
)
def test_scenario_group2(
    span_evaluator,
    scenario,
    annotation,
    prediction,
    tokens,
    start_indices,
    expected_tp,
    expected_fp,
    expected_fn,
    expected_results,
    expected_error_types,
):
    """Test each scenario in Group 2: annotated spans overlapping with multiple prediction spans."""
    # Build the DataFrame expected by SpanEvaluator
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "prediction": prediction,
            "start_indices": start_indices,
        }
    )

    # Run evaluation
    result = EvaluationResult()
    result = span_evaluator.calculate_score_on_df(
        per_type=True, results_df=df, evaluation_result=result
    )

    # Check true positives, false positives, and false negatives
    total_tp = sum(pii_type.true_positives for pii_type in result.per_type.values())
    total_fp = sum(pii_type.false_positives for pii_type in result.per_type.values())
    total_fn = sum(pii_type.false_negatives for pii_type in result.per_type.values())

    assert total_tp == expected_tp, f"In {scenario}, expected {expected_tp} TPs, got {total_tp}"
    assert total_fp == expected_fp, f"In {scenario}, expected {expected_fp} FPs, got {total_fp}"
    assert total_fn == expected_fn, f"In {scenario}, expected {expected_fn} FNs, got {total_fn}"

    assert_confusion_matrix(expected_results, result, scenario)

    assert_error_types(expected_error_types, result, scenario)




# Test global PII evaluatiom metrics

@pytest.mark.parametrize(
    "scenario, annotation, prediction, tokens, start_indices, expected_precision, expected_recall, expected_f",
    [
        # Perfect match
        (
            "Scenario 8: Perfect match",
            ["O", "PERSON", "PERSON", "O"],
            ["O", "LOCATION", "PERSON", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            1.0,  # precision
            1.0,  # recall
            1.0,  # F1 score
        ),
        # No matches
        (
            "Scenario 9: No matches",
            ["O", "PERSON", "ORG", "O"],
            ["O", "O", "O", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            np.nan,  # precision
            0.0,  # recall
            np.nan,  # F1 score
        ),
        # One match out of two
        (
            "Scenario 10: Partial match",
            ["O", "PERSON", "O", "CAT"],
            ["O", "PERSON", "O", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            1.0,  # precision
            0.5,  # recall
            0.5555,  # F1 score
        ),
    ],
)
def test_global_metrics(
    span_evaluator,
    scenario,
    annotation,
    prediction,
    tokens,
    start_indices,
    expected_precision,
    expected_recall,
    expected_f,
):
    """Test global PII evaluation metrics."""
    # Build the DataFrame expected by SpanEvaluator
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "prediction": prediction,
            "start_indices": start_indices,
        }
    )

    # Run evaluation
    result = EvaluationResult()
    pii_df = span_evaluator.create_global_entities_df(df)
    result = span_evaluator.calculate_score_on_df(
        per_type=False, results_df=pii_df, evaluation_result=result
    )

    # Check global metrics
    assert_metric(expected_precision, "precision", result, scenario)
    assert_metric(expected_recall, "recall", result, scenario)
    assert_metric(expected_f, "f", result, scenario)



# Test Both per_type and global metrics together
@pytest.mark.parametrize(
    "scenario, annotation, prediction, tokens, start_indices, expected_per_type_metrics, expected_global_metrics",
    [
        # Perfect detection scenario
        (
            "Perfect detection: all entities found correctly",
            ["PERSON", "PERSON", "O", "O", "LOCATION", "O"],
            ["PERSON", "PERSON", "O", "O", "LOCATION", "O"],
            ["John", "Smith", "lives", "in", "Boston", "today"],
            [0, 5, 11, 17, 20, 27],
            {
                "PERSON": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "LOCATION": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            },
            {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        ),
        # Address partially detected, with multiple pred spans per annotated span
        (
            "Address partially detected: street number found but street name missed",
            ["PERSON", "O", "O", "ADDRESS", "ADDRESS", "ADDRESS", "O"],
            ["PERSON", "O", "O", "ADDRESS", "O", "ADDRESS", "O"],
            ["Alice", "lives", "at", "123", "Main", "Street", "downtown"],
            [0, 6, 12, 15, 19, 24, 31],
            {
                "PERSON": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "ADDRESS": {"precision": 1.0, "recall": 0.3333333333333333, "f1": 0.5},
            },
            {"precision": 1.0, "recall": 0.6666666666666666, "f1": 0.8},
        ),
        # Mixed entity type confusion
        (
            "Entity type confusion: location detected as person",
            ["PERSON", "PERSON", "O", "O", "LOCATION", "O"],
            ["PERSON", "PERSON", "O", "O", "PERSON", "O"],
            ["Bob", "Davis", "went", "to", "Chicago", "yesterday"],
            [0, 4, 10, 15, 18, 26],
            {
                "PERSON": {"precision": 0.6666666666666666, "recall": 1.0, "f1": 0.8},
                "LOCATION": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            },
            {"precision": 0.6666666666666666, "recall": 1.0, "f1": 0.8},
        ),
        # Complete miss for one entity type
        (
            "Complete miss: phone number not detected at all",
            ["PERSON", "O", "PHONE_NUMBER", "PHONE_NUMBER", "O"],
            ["PERSON", "O", "O", "O", "O"],
            ["Sarah", "called", "555", "1234", "today"],
            [0, 6, 13, 17, 22],
            {
                "PERSON": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "PHONE_NUMBER": {"precision": np.nan, "recall": 0.0, "f1": np.nan},
            },
            {"precision": 1.0, "recall": 0.5, "f1": 0.6666666666666666},
        ),
        # False positive scenario
        (
            "False positive: common word detected as entity",
            ["O", "PERSON", "O", "O", "O", "O"],
            ["O", "PERSON", "O", "LOCATION", "O", "O"],
            ["The", "president", "spoke", "about", "security", "issues"],
            [0, 4, 14, 20, 26, 35],
            {
                "PERSON": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "LOCATION": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            },
            {"precision": 0.5, "recall": 1.0, "f1": 0.6666666666666666},
        ),
        # No entities detected
        (
            "No detection: all entities missed",
            ["PERSON", "PERSON", "O", "LOCATION", "O"],
            ["O", "O", "O", "O", "O"],
            ["Emma", "Thompson", "visited", "Paris", "today"],
            [0, 5, 14, 22, 28],
            {
                "PERSON": {"precision": np.nan, "recall": 0.0, "f1": np.nan},
                "LOCATION": {"precision": np.nan, "recall": 0.0, "f1": np.nan},
            },
            {"precision": np.nan, "recall": 0.0, "f1": np.nan},
        ),
    ],
)
def test_combined_per_type_and_global_metrics(
    span_evaluator,
    scenario,
    annotation,
    prediction,
    tokens,
    start_indices,
    expected_per_type_metrics,
    expected_global_metrics,
):
    """Test that per-type and global metrics are calculated correctly together."""
    # Build the DataFrame expected by SpanEvaluator
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "prediction": prediction,
            "start_indices": start_indices,
        }
    )

    # Run per-type evaluation
    result = EvaluationResult()
    result = span_evaluator.calculate_score_on_df(
        per_type=True, results_df=df, evaluation_result=result
    )

    # Check per-type metrics
    for entity_type, expected_metrics in expected_per_type_metrics.items():
        if entity_type in result.per_type:
            per_type_result = result.per_type[entity_type]
            if not np.isnan(expected_metrics["precision"]):
                assert per_type_result.precision == pytest.approx(expected_metrics["precision"], abs=1e-3), (
                    f"In {scenario}, {entity_type} precision expected {expected_metrics['precision']}, got {per_type_result.precision}"
                )
            else:
                assert np.isnan(per_type_result.precision), f"In {scenario}, {entity_type} precision should be NaN"
            
            if not np.isnan(expected_metrics["recall"]):
                assert per_type_result.recall == pytest.approx(expected_metrics["recall"], abs=1e-3), (
                    f"In {scenario}, {entity_type} recall expected {expected_metrics['recall']}, got {per_type_result.recall}"
                )
            else:
                assert np.isnan(per_type_result.recall), f"In {scenario}, {entity_type} recall should be NaN"

    # Run global evaluation
    pii_df = span_evaluator.create_global_entities_df(df)
    result = span_evaluator.calculate_score_on_df(
        per_type=False, results_df=pii_df, evaluation_result=result
    )

    # Check global metrics
    assert_metric(expected_global_metrics["precision"], "precision", result, scenario)
    assert_metric(expected_global_metrics["recall"], "recall", result, scenario)
    assert_metric(expected_global_metrics["f1"], "f", result, scenario)


# TODO: Test per-token IoU

# TODO: Test error analysis

# TODO: Test span creation and skip words handling
