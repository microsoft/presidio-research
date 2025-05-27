import pytest
import pandas as pd
from presidio_evaluator.evaluation.span_evaluator import SpanEvaluator
from presidio_evaluator.data_objects import Span


@pytest.mark.parametrize(
    "annotation, prediction, tokens, start_indices, TP, num_annotated, num_predicted",
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
        ),  # Annotated: "John Doe", Predicted: "John" and "Doe" separately
        # No Overlap: Completely Different Entities
        (
            ["PERSON", "O", "O", "O"],
            ["LOCATION", "O", "O", "O"],
            ["Paris", "is", "beautiful", "today"],
            [True, False, False, False],
            0,
            1,
            1,
        ),  # Annotated: "Paris" as PERSON, Predicted: "Paris" as LOCATION
        # One Entity: One Correct, One Incorrect
        (
                ["PERSON", "O", "O", "PERSON"],
                ["PERSON", "O", "PERSON", "PERSON"],
                ["Alice", "went", "to", "Paris"],
                [True, False, False, True],
                2,
                2,
                2,
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
        ),  # Annotated: "Sir Arthur Conan Doyle", Predicted: "Sir" and "Conan Doyle"
        # Multiple Predicted Spans for a Single Annotated Span
        (
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["PERSON", "O", "PERSON", "O"],
            ["Marie", "Claire", "de", "Roth"],
            [True, False, False, False],
            0,
            1,
            2,
        ),  # Annotated: "Marie Claire de Roth", Predicted: "Marie" and "de"
        # No Entities in Annotation, Some in Prediction
        (
            ["O", "O", "O", "O"],
            ["PERSON", "O", "LOCATION", "O"],
            ["This", "is", "London", "now"],
            [False, False, False, False],
            0,
            0,
            2,
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
        ),  # Prediction is shorter than annotation
        (
            ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "PERSON", "PERSON", "PERSON"],
            ["John", "Doe", "and", "Jane", "Smith"],
            [True, False, False, True, False],
            0,
            2,
            1,
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
        ),
    ],
)
def test_evaluate(
    annotation, prediction, tokens, start_indices, TP, num_annotated, num_predicted
):
    # Build the DataFrame expected by SpanEvaluator
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "prediction": prediction,
            "is_entity_start": start_indices,
        }
    )
    print(f"df: {df}")
    evaluator = SpanEvaluator(iou_threshold=0.9, schema=None)
    result = evaluator.evaluate(df)

    # Calculate expected metrics
    expected_recall = TP / num_annotated if num_annotated > 0 else 0
    expected_precision = TP / num_predicted if num_predicted > 0 else 0

    assert result["recall"] == pytest.approx(
        expected_recall
    ), f"Recall mismatch: expected {expected_recall}, got {result['recall']}"
    assert result["precision"] == pytest.approx(
        expected_precision
    ), f"Precision mismatch: expected {expected_precision}, got {result['precision']}"


def test_evaluate_with_custom_skipwords():
    df = pd.DataFrame(
        {
            "sentence_id": [0] * 5,
            "token": ["David", "paid", "the", "bill", "today"],
            "annotation": ["PERSON", "O", "O", "O", "O"],
            "prediction": ["PERSON", "O", "O", "PERSON", "O"],
            "is_entity_start": [True, False, False, False, False],
        }
    )

    evaluator = SpanEvaluator(iou_threshold=0.9, schema=None, skip_words=["bill"])
    result = evaluator.evaluate(df)

    # Calculate expected metrics
    expected_recall = 1
    expected_precision = 1

    assert result["recall"] == pytest.approx(expected_recall)
    assert result["precision"] == pytest.approx(expected_precision)


@pytest.mark.parametrize(
    "tokens, bio_labels, expected_spans",
    [
        # Simple single entity
        (
            ["John", "Smith", "is", "here"],
            ["B-PERSON", "I-PERSON", "O", "O"],
            [
                Span(
                    entity_type="PERSON",
                    entity_value=["John", "Smith"],
                    start_position=0,
                    end_position=2,
                )
            ],
        ),
        # Multiple entities
        (
            ["John", "Smith", "at", "Microsoft", "Corp"],
            ["B-PERSON", "I-PERSON", "O", "B-ORGANIZATION", "I-ORGANIZATION"],
            [
                Span(
                    entity_type="PERSON",
                    entity_value=["John", "Smith"],
                    start_position=0,
                    end_position=2,
                ),
                Span(
                    entity_type="ORGANIZATION",
                    entity_value=["Microsoft", "Corp"],
                    start_position=3,
                    end_position=5,
                ),
            ],
        ),
        # Single token entity
        (
            ["John", "went", "to", "Paris"],
            ["B-PERSON", "O", "O", "B-LOCATION"],
            [
                Span(
                    entity_type="PERSON",
                    entity_value=["John"],
                    start_position=0,
                    end_position=1,
                ),
                Span(
                    entity_type="LOCATION",
                    entity_value=["Paris"],
                    start_position=3,
                    end_position=4,
                ),
            ],
        ),
        # No entities
        (["The", "sky", "is", "blue"], ["O", "O", "O", "O"], []),
        # Invalid I- without B-
        (["John", "Smith", "is", "here"], ["I-PERSON", "I-PERSON", "O", "O"], []),
    ],
)
def test_create_spans_bio(tokens, bio_labels, expected_spans):
    df = pd.DataFrame(
        {"sentence_id": [0] * len(tokens), "token": tokens, "annotation": bio_labels}
    )

    evaluator = SpanEvaluator(iou_threshold=0.9, schema="BIO")
    spans = evaluator._create_spans_bio(df, "annotation")

    assert len(spans) == len(
        expected_spans
    ), f"Number of spans mismatch. Expected {len(expected_spans)}, got {len(spans)}"

    for span, expected_span in zip(spans, expected_spans):
        assert (
            span.entity_type == expected_span.entity_type
        ), f"Entity type mismatch. Expected {expected_span.entity_type}, got {span.entity_type}"
        assert (
            span.entity_value == expected_span.entity_value
        ), f"Entity value mismatch. Expected {expected_span.entity_value}, got {span.entity_value}"
        assert (
            span.start_position == expected_span.start_position
        ), f"Start position mismatch. Expected {expected_span.start_position}, got {span.start_position}"
        assert (
            span.end_position == expected_span.end_position
        ), f"End position mismatch. Expected {expected_span.end_position}, got {span.end_position}"