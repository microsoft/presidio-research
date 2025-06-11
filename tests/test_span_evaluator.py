import pytest
import pandas as pd
from presidio_evaluator.data_objects import Span
from presidio_evaluator.evaluation.span_evaluator import SpanEvaluator
from presidio_evaluator.evaluation.evaluation_result import EvaluationResult
from tests.mocks import MockModel


@pytest.mark.parametrize(
    "annotation, prediction, tokens, start_indices, TP, num_annotated, num_predicted, char_based",
    [
        # Single Entity with a Skip Word
        (
            ["PERSON", "O", "O", "O"],
            ["PERSON", "O", "O", "O"],
            ["David", "is", "my", "friend"],
            [True, False, False, False],
            1,
            1,
            1,
            False,
        ),  # 'is' is a skip word
        # Predicted entity with a skip word gap
        (
            ["PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "O", "PERSON", "O"],
            ["David", "is", "living", "abroad"],
            [True, False, False, False],
            1,
            1,
            1,
            False,
        ),  # 'is' is a skip word
        # Annotated entity with a punctuation gap
        (
            ["PERSON", "O", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["David", ",", "Maxwell", "Morris"],
            [True, False, True, False],
            1,
            1,
            1,
            False,
        ),  # ',' is a punctuation character
        # Predicted entity with a punctuation gap
        (
            ["PERSON", "O", "O", "PERSON"],
            ["PERSON", "O", "O", "PERSON"],
            ["David", "-", "-", "Morris"],
            [True, False, False, True],
            1,
            1,
            1,
            False,
        ),  # '-' is a punctuation character
        # End of entity skip words
        (
            ["PERSON", "PERSON", "O", "O", "O"],
            ["PERSON", "PERSON", "PERSON", "PERSON", "O"],
            ["David", "Johnson", "is", "my", "friend"],
            [True, False, False, False, False],
            1,
            1,
            1,
            False,
        ),
        # Prediction Misses Entire Annotated Span
        (
            ["PERSON", "PERSON", "O", "O"],
            ["O", "O", "O", "O"],
            ["David", "Johnson", "my", "friend"],
            [True, False, False, False],
            0,
            1,
            0,
            False,
        ),
        # Partial Overlap: Start Boundary Mismatch
        (
            ["LOCATION", "LOCATION", "LOCATION", "O"],
            ["O", "LOCATION", "LOCATION", "O"],
            ["New", "York", "City", "is"],
            [True, False, False, False],
            0,
            1,
            1,
            False,
        ),  # Annotated: "New York City", Predicted: "York City"
        # Partial Overlap: End Boundary Mismatch
        (
            ["O", "O", "PERSON", "PERSON"],
            ["O", "PERSON", "O", "PERSON"],
            ["I", "met", "John", "Doe"],
            [False, False, True, False],
            0,
            1,
            2,
            False,
        ),  # Annotated: "John Doe", Predicted: "John" and "Doe" separately
        # No Overlap: Completely Different Entities
        (
            ["PERSON", "O", "O", "O"],
            ["LOCATION", "O", "O", "O"],
            ["Paris", "is", "beautiful", "today"],
            [True, False, False, False],
            1,
            1,
            1,
            False,
        ),  # Annotated: "Paris" as PERSON, Predicted: "Paris" as LOCATION
        # One Entity: One Correct, One Incorrect
        (
            ["PERSON", "O", "O", "PERSON"],
            ["PERSON", "O", "PERSON", "PERSON"],
            ["Alice", "went", "to", "Paris"],
            [True, False, False, True],
            1,
            2,
            2,
            False,
        ),
        # Multiple Entities: One Correct, One Incorrect
        (
            ["PERSON", "O", "O", "LOCATION"],
            ["PERSON", "O", "PERSON", "PERSON"],
            ["Alice", "went", "to", "Paris"],
            [True, False, False, True],
            1,
            2,
            2,
            False,
        ),  # "Alice" correctly predicted, "Paris" mispredicted as PERSON
        # Overlapping Entities: Nested Span (Not Typical in NER but for Robustness)
        (
            ["PERSON", "PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "O", "PERSON", "PERSON", "O"],
            ["Sir", "Arthur", "Conan", "Doyle", "wrote"],
            [True, False, False, False, False],
            0,
            1,
            2,
            False,
        ),  # Annotated: "Sir Arthur Conan Doyle", Predicted: "Sir" and "Conan Doyle"
        # Multiple Predicted Spans for a Single Annotated Span
        (
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["PERSON", "O", "PERSON", "O"],
            ["Marie", "Claire", "de", "Roth"],
            [True, False, False, False],
            0,
            1,
            1,
            False,
        ),  # Annotated: "Marie Claire de Roth", Predicted: "Marie" and "de"
        # No Entities in Annotation, Some in Prediction
        (
            ["O", "O", "O", "O"],
            ["PERSON", "O", "LOCATION", "O"],
            ["This", "is", "London", "now"],
            [False, False, False, False],
            0,
            0,
            1,
            False,
        ),  # All predictions are false positives
        # No Entities in Prediction, Some in Annotation with a skip word
        (
            ["O", "O", "O", "O"],
            ["PERSON", "O", "PERSON", "O"],
            ["This", "is", "London", "now"],
            [False, False, False, False],
            0,
            0,
            1,
            False,
        ),  # All predictions are false positives but merge into one entity
        # No Entities in Prediction, Some in Annotation
        (
            ["PERSON", "O", "LOCATION", "LOCATION"],
            ["O", "O", "O", "O"],
            ["Emma", "travels", "to", "Berlin"],
            [True, False, True, False],
            0,
            2,
            0,
            False,
        ),  # All annotations are false negatives
        # Exact Match with Multiple Entities
        (
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["Barack", "Obama", "visited", "New", "York"],
            [True, False, False, True, False],
            2,
            2,
            2,
            False,
        ),  # Two entities correctly predicted
        # Exact Match with Multiple Entities and skip word
        (
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["George", "Washington", "is", "George", "Washington"],
            [True, False, False, True, False],
            2,
            2,
            2,
            False,
        ),  # Two entities correctly predicted
        # Adjacent Entities Without Overlap
        (
            ["PERSON", "PERSON", "LOCATION", "LOCATION"],
            ["PERSON", "PERSON", "LOCATION", "LOCATION"],
            ["John", "Doe", "Paris", "France"],
            [True, False, True, False],
            2,
            2,
            2,
            False,
        ),  # Two adjacent entities correctly predicted
        # Prediction Extends Beyond Annotated Span
        (
            ["PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["Anna", "Marie", "Smith", "Loves"],
            [True, False, False, False],
            0,
            1,
            1,
            False,
        ),  # Prediction merges into a single span, but is longer than the annotated span
        # Prediction Extends Beyond Annotated Span
        (
            ["PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["John", "Doe", "Smith", "son"],
            [True, False, False, False],
            0,
            1,
            1,
            False,
        ),  # Prediction is longer than annotation
        # Prediction is Subset of Annotated Span
        (
            ["PERSON", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "O"],
            ["John", "Doe", "Smith"],
            [True, False, False],
            0,
            1,
            1,
            False,
        ),  # Prediction is shorter than annotation
        (
            ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "PERSON", "PERSON", "PERSON"],
            ["John", "Doe", "and", "Jane", "Smith"],
            [True, False, False, True, False],
            1,
            1,
            1,
            False,
        ),  # Annotated: "John Doe" and "Jane Smith" separately, Predicted:
        # "John Doe and Jane Smith" as one merged entity
        (
            ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "PERSON", "PERSON", "PERSON"],
            ["John", "Doe", "and", "Jane", "Smith"],
            [True, False, False, True, False],
            0,
            2,
            1,
            False,
        ),  # Annotated: "John Doe" and "Jane Smith" separately, Predicted:
        # "John Doe and Jane Smith" as one merged entity
        (
            ["PERSON", "PERSON", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
            ["PERSON", "PERSON", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
            ["O'Brien", "Jr.", "at", "McDonald's", "Corp.", "Inc."],
            [True, False, False, True, False, False],
            2,
            2,
            2,
            False,
        ),  # Special characters test: "O'Brien Jr." as PERSON and "McDonald's Corp. Inc."
        # as ORGANIZATION
        (
            ["O"] * 1000,
            ["O"] * 1000,
            ["word"] * 1000,
            [False] * 1000,
            0,
            0,
            0,
            False,
        ),
        # Adjacent entities of same type - not merged when merge_adjacent_spans=False
        (
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["John", "Doe", "Smith", "Jr"],
            [True, False, True, False],
            1,
            1,
            1,
            False,
        ),
        # Adjacent entities with intervening 'O' token - should behave the same with merge_adjacent_spans=False
        (
            ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
            ["John", "Doe", "and", "Jane", "Smith"],
            [True, False, False, True, False],
            2,
            2,
            2,
            False,
        ),
        # Partially overlapping predictions - counted separately when merge_adjacent_spans=False
        (
            ["PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "O", "PERSON", "PERSON"],
            ["James", "Robert", "Smith", "III"],
            [True, False, True, False],
            0,
            1,
            2,
            False,
        ),
        # Multiple adjacent predictions matching single annotation - counted as separate when merge_adjacent_spans=False
        (
            ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
            ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
            ["United", "-", "States", "Government"],
            [True, False, False, True],
            1,
            1,
            1,
            False,
        ),
        # Char-based: Single entity exact match
        (
            ["PERSON", "PERSON", "O", "O"],
            ["PERSON", "PERSON", "O", "O"],
            ["John", "Smith", "works", "here"],
            [True, False, False, False],
            1,
            1,
            1,
            True,
        ),
        # Char-based: Multiple entities exact match
        (
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["John", "Smith", "visited", "New", "York"],
            [True, False, False, True, False],
            2,
            2,
            2,
            True,
        ),
        # Char-based: Partial overlap at character level
        (
            ["PERSON", "PERSON", "O", "O"],
            ["PERSON", "O", "O", "O"],
            ["John", "Smith", "is", "here"],
            [True, False, False, False],
            0,  # No exact match at char level
            1,
            1,
            True,
        ),
        # Char-based: Entity with punctuation
        (
            ["PERSON", "O", "PERSON", "O"],
            ["PERSON", "O", "PERSON", "O"],
            ["Dr.", ",", "Smith", "arrived"],
            [True, False, True, False],
            1,
            1,
            1,
            True,
        ),
        # Char-based: Skip words merging entities
        (
            ["PERSON", "O", "PERSON", "O"],
            ["PERSON", "O", "PERSON", "O"],
            ["John", "and", "Mary", "came"],
            [True, False, True, False],
            1,  # Merged into single entity due to skip word
            1,
            1,
            True,
        ),
        # Char-based: No merge adjacent spans
        (
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["John", "F.", "Kennedy", "Jr."],
            [True, False, True, False],
            1,
            1,
            1,
            True,
        ),
        # Char-based: Different entity types at same position
        (
            ["PERSON", "O", "O", "O"],
            ["LOCATION", "O", "O", "O"],
            ["Paris", "is", "beautiful", "today"],
            [True, False, False, False],
            1,  # Different entity types
            1,
            1,
            True,
        ),
        # Char-based: Prediction extends beyond annotation
        (
            ["PERSON", "PERSON", "O", "O"],
            ["PERSON", "PERSON", "PERSON", "O"],
            ["John", "Smith", "Jr.", "arrived"],
            [True, False, True, False],
            0,  # Prediction longer than annotation
            1,
            1,
            True,
        ),
        # Char-based: Annotation extends beyond prediction
        (
            ["PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "PERSON", "O", "O"],
            ["John", "F.", "Kennedy", "arrived"],
            [True, False, True, False],
            0,  # Annotation longer than prediction
            1,
            1,
            True,
        ),
        # Char-based: Multiple short entities
        (
            ["PERSON", "O", "PERSON", "O", "LOCATION"],
            ["PERSON", "O", "PERSON", "O", "LOCATION"],
            ["John", "met", "Jane", "in", "Paris"],
            [True, False, True, False, True],
            3,
            3,
            3,
            True,
        ),
        # Char-based: Empty predictions
        (
            ["PERSON", "PERSON", "O", "O"],
            ["O", "O", "O", "O"],
            ["John", "Smith", "works", "here"],
            [True, False, False, False],
            0,
            1,
            0,
            True,
        ),
        # Char-based: Empty annotations
        (
            ["O", "O", "O", "O"],
            ["PERSON", "PERSON", "O", "O"],
            ["John", "Smith", "works", "here"],
            [True, False, False, False],
            0,
            0,
            1,
            True,
        ),
        # Char-based: Complex case with multiple overlaps
        (
            ["PERSON", "PERSON", "O", "ORGANIZATION"],
            ["PERSON", "O", "PERSON", "ORGANIZATION"],
            ["John", "Smith", "at", "Microsoft"],
            [True, False, False, True],
            1,  # Only "Microsoft" matches exactly
            2,
            2,
            True,
        ),
    ],
)
def test_evaluate(
    annotation,
    prediction,
    tokens,
    start_indices,
    TP,
    num_annotated,
    num_predicted,
    char_based,
):
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
    print(f"df: {df}")
    evaluator = SpanEvaluator(
        model=MockModel(),
        iou_threshold=0.9,
        char_based=char_based,
    )
    result = evaluator.calculate_score_on_df(df)

    # Calculate expected metrics
    expected_recall = TP / num_annotated if num_annotated > 0 else 0
    expected_precision = TP / num_predicted if num_predicted > 0 else 0

    # Assert that the result is an SpanEvaluationResult instance
    assert isinstance(result, EvaluationResult)

    # Check counts
    assert result.total_predicted == num_predicted
    assert result.total_annotated == num_annotated
    assert result.total_true_positives == TP

    assert result.pii_precision == pytest.approx(
        expected_precision
    ), f"Precision mismatch: expected {expected_precision}, got {result.pii_precision}"
    assert result.pii_recall == pytest.approx(
        expected_recall
    ), f"Recall mismatch: expected {expected_recall}, got {result.pii_recall}"


def test_evaluate_with_custom_skipwords():
    df = pd.DataFrame(
        {
            "sentence_id": [0] * 5,
            "token": ["David", "paid", "the", "bill", "today"],
            "annotation": ["PERSON", "O", "O", "O", "O"],
            "prediction": ["PERSON", "O", "O", "PERSON", "O"],
            "start_indices": [True, False, False, False, False],
        }
    )

    evaluator = SpanEvaluator(model=MockModel(), iou_threshold=0.9, skip_words=["bill"])
    result = evaluator.calculate_score_on_df(df)

    # Calculate expected metrics
    expected_recall = 1
    expected_precision = 1

    # Assert that the result is an SpanEvaluationResult instance
    assert isinstance(result, SpanEvaluationResult)

    assert result.recall == pytest.approx(
        expected_recall
    ), f"Recall mismatch: expected {expected_recall}, got {result.recall}"
    assert result.precision == pytest.approx(
        expected_precision
    ), f"Precision mismatch: expected {expected_precision}, got {result.precision}"
