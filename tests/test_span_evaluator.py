import pytest
import pandas as pd
from presidio_evaluator.data_objects import Span
from presidio_evaluator.evaluation.span_evaluator import SpanEvaluator
from presidio_evaluator.evaluation.evaluation_result import EvaluationResult
from tests.mocks import MockModel


@pytest.mark.parametrize(
    "annotation, prediction, tokens, start_indices, TP, num_annotated, num_predicted, char_based",
    [
        # BASIC ENTITY MATCHING
        (
            ["PERSON", "O", "O", "O"],
            ["PERSON", "O", "O", "O"],
            ["David", "is", "my", "friend"],
            [0, 6, 9, 12],
            1, 1, 1, False
        ),
        (
            ["EMAIL", "O", "O", "O"],
            ["EMAIL", "O", "O", "O"],
            ["user@example.com", "sent", "a", "message"],
            [0, 17, 22, 24],
            1, 1, 1, False
        ),
        
        # SKIP WORD HANDLING
        (
            ["PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "O", "PERSON", "O"],
            ["David", "is", "living", "abroad"],
            [0, 6, 9, 16],
            1, 1, 1, False
        ),
        (
            ["PERSON", "O", "PERSON", "O"],
            ["PERSON", "O", "PERSON", "O"],
            ["John", "and", "Mary", "came"],
            [0, 5, 9, 14],
            1, 1, 1, True
        ),
        (
            ["PERSON", "O", "O", "O", "PERSON", "O"],
            ["PERSON", "O", "O", "O", "PERSON", "O"],
            ["John", "and", "the", "other", "Smith", "arrived"],
            [0, 5, 9, 13, 19, 25],
            1, 1, 1, False
        ),
        
        # PUNCTUATION HANDLING
        (
            ["PERSON", "O", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["David", ",", "Maxwell", "Morris"],
            [0, 6, 8, 16],
            1, 1, 1, False
        ),
        (
            ["PERSON", "O", "O", "PERSON"],
            ["PERSON", "O", "O", "PERSON"],
            ["David", "-", "-", "Morris"],
            [0, 6, 8, 10],
            1, 1, 1, False
        ),
        (
            ["PERSON", "O", "PERSON", "O"],
            ["PERSON", "O", "PERSON", "O"],
            ["Dr.", ",", "Smith", "arrived"],
            [0, 4, 6, 12],
            1, 1, 1, True
        ),
        (
            ["PERSON", "O", "O", "O", "PERSON"],
            ["PERSON", "O", "O", "O", "PERSON"],
            ["James", ".", "-", "/", "Bond"],
            [0, 6, 8, 10, 12],
            1, 1, 1, False
        ),
        
        # BOUNDARY MISMATCHES
        (
            ["LOCATION", "LOCATION", "LOCATION", "O"],
            ["O", "LOCATION", "LOCATION", "O"],
            ["New", "York", "City", "is"],
            [0, 4, 9, 14],
            0, 1, 1, False
        ),
        (
            ["O", "O", "PERSON", "PERSON"],
            ["O", "PERSON", "O", "PERSON"],
            ["I", "met", "John", "Doe"],
            [0, 2, 6, 11],
            0, 1, 2, False
        ),
        (
            ["PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["Anna", "Marie", "Smith", "Loves"],
            [0, 5, 11, 17],
            0, 1, 1, False
        ),
        (
            ["PERSON", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "O"],
            ["John", "Doe", "Smith"],
            [0, 5, 9],
            0, 1, 1, False
        ),
        (
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "PERSON", "PERSON", "O", "O"],
            ["Mr", "John", "Middle", "Smith", "Jr"],
            [0, 3, 8, 15, 21],
            0, 1, 1, False
        ),
        
        # ENTITY TYPE MISMATCHES
        (
            ["PERSON", "O", "O", "O"],
            ["LOCATION", "O", "O", "O"],
            ["Paris", "is", "beautiful", "today"],
            [0, 6, 9, 19],
            1, 1, 1, False
        ),
        (
            ["CREDIT_CARD", "CREDIT_CARD", "CREDIT_CARD", "CREDIT_CARD"],
            ["PHONE_NUMBER", "PHONE_NUMBER", "PHONE_NUMBER", "PHONE_NUMBER"],
            ["1234", "5678", "9012", "3456"],
            [0, 5, 10, 15],
            1, 1, 1, False
        ),
        
        # MULTIPLE ENTITY SCENARIOS
        (
            ["PERSON", "O", "O", "LOCATION"],
            ["PERSON", "O", "PERSON", "PERSON"],
            ["Alice", "went", "to", "Paris"],
            [0, 6, 11, 14],
            2, 2, 2, False
        ),
        (
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["Barack", "Obama", "visited", "New", "York"],
            [0, 7, 13, 21, 25],
            2, 2, 2, False
        ),
        (
            ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
            ["John", "Doe", "and", "Jane", "Smith"],
            [0, 5, 9, 13, 18],
            1, 1, 1, False
        ),
        (
            ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "PERSON", "PERSON", "PERSON"],
            ["John", "Doe", "and", "Jane", "Smith"],
            [0, 5, 9, 13, 18],
            1, 1, 1, False
        ),
        (
            ["PERSON", "O", "PERSON", "O", "LOCATION"],
            ["PERSON", "O", "PERSON", "O", "LOCATION"],
            ["John", "met", "Jane", "in", "Paris"],
            [0, 5, 9, 14, 17],
            3, 3, 3, True
        ),
        (
            ["PERSON", "PERSON", "O", "LOCATION", "O", "DATE", "DATE"],
            ["PERSON", "PERSON", "O", "ORGANIZATION", "O", "DATE", "O"],
            ["John", "Smith", "visited", "London", "on", "January", "1st"],
            [0, 5, 11, 19, 26, 29, 37],
            2, 3, 3, False
        ),
        
        # OVERLAPPING/NESTED ENTITIES
        (
            ["PERSON", "PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "O", "PERSON", "PERSON", "O"],
            ["Sir", "Arthur", "Conan", "Doyle", "wrote"],
            [0, 4, 11, 17, 23],
            0, 1, 2, False
        ),
        (
            ["PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "O", "PERSON", "PERSON"],
            ["James", "Robert", "Smith", "III"],
            [0, 6, 13, 19],
            0, 1, 2, False
        ),
        
        # SPECIAL CHARACTERS AND FORMATTING
        (
            ["PERSON", "PERSON", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
            ["PERSON", "PERSON", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
            ["O'Brien", "Jr.", "at", "McDonald's", "Corp.", "Inc."],
            [0, 8, 12, 15, 27, 34],
            2, 2, 2, False
        ),
        (
            ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
            ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
            ["United", "-", "States", "Government"],
            [0, 7, 9, 16],
            1, 1, 1, False
        ),
        (
            ["PERSON", "PERSON", "O", "O"],
            ["PERSON", "PERSON", "O", "O"],
            ["José", "Martínez", "from", "España"],
            [0, 5, 14, 19],
            1, 1, 1, True
        ),
        (
            ["ID_NUMBER", "ID_NUMBER", "ID_NUMBER"],
            ["ID_NUMBER", "ID_NUMBER", "ID_NUMBER"],
            ["ID", "-", "12345"],
            [0, 3, 5],
            1, 1, 1, False
        ),
        
        # EDGE CASES
        (
            ["O"] * 1000,
            ["O"] * 1000,
            ["word"] * 1000,
            list(range(0, 5000, 5)),  # Start positions spaced by 5 characters
            0, 0, 0, False
        ),
        (
            ["O", "O", "O", "O"],
            ["PERSON", "O", "LOCATION", "O"],
            ["This", "is", "London", "now"],
            [0, 5, 8, 15],
            0, 0, 1, False
        ),
        (
            ["PERSON", "O", "LOCATION", "LOCATION"],
            ["O", "O", "O", "O"],
            ["Emma", "travels", "to", "Berlin"],
            [0, 5, 13, 16],
            0, 2, 0, False
        ),
        (
            ["PERSON", "PERSON", "O", "O"],
            ["O", "O", "O", "O"],
            ["John", "Smith", "works", "here"],
            [0, 5, 11, 17],
            0, 1, 0, True
        ),
        (
            ["O", "O", "O", "O"],
            ["PERSON", "PERSON", "O", "O"],
            ["John", "Smith", "works", "here"],
            [0, 5, 11, 17],
            0, 0, 1, True
        ),
        (
            ["LOCATION", "O", "LOCATION", "O", "LOCATION"],
            ["LOCATION", "O", "LOCATION", "O", "LOCATION"],
            ["UK", "and", "US", "and", "EU"],
            [0, 3, 7, 10, 14],
            1, 1, 1, False
        ),
        
        # CHARACTER-BASED EVALUATION
        (
            ["PERSON", "PERSON", "O", "O"],
            ["PERSON", "PERSON", "O", "O"],
            ["John", "Smith", "works", "here"],
            [0, 5, 11, 17],
            1, 1, 1, True
        ),
        (
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["John", "Smith", "visited", "New", "York"],
            [0, 5, 11, 19, 23],
            2, 2, 2, True
        ),
        (
            ["PERSON", "PERSON", "O", "O"],
            ["PERSON", "O", "O", "O"],
            ["John", "Smith", "is", "here"],
            [0, 5, 11, 14],
            0, 1, 1, True
        ),
        (
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["John", "F.", "Kennedy", "Jr."],
            [0, 5, 8, 16],
            1, 1, 1, True
        ),
        (
            ["PERSON", "PERSON", "O", "O"],
            ["PERSON", "PERSON", "PERSON", "O"],
            ["John", "Smith", "Jr.", "arrived"],
            [0, 5, 11, 15],
            0, 1, 1, True
        ),
        (
            ["PERSON", "PERSON", "PERSON", "O"],
            ["PERSON", "PERSON", "O", "O"],
            ["John", "F.", "Kennedy", "arrived"],
            [0, 5, 8, 16],
            0, 1, 1, True
        ),
        (
            ["PERSON", "PERSON", "O", "ORGANIZATION"],
            ["PERSON", "O", "PERSON", "ORGANIZATION"],
            ["John", "Smith", "at", "Microsoft"],
            [0, 5, 11, 14],
            1, 2, 2, True
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

    # Assert that the result is an EvaluationResult instance
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
            "start_indices": [0, 6, 11, 15, 20],
        }
    )

    evaluator = SpanEvaluator(model=MockModel(), iou_threshold=0.9, skip_words=["bill"])
    result = evaluator.calculate_score_on_df(df)

    # Calculate expected metrics
    expected_recall = 1
    expected_precision = 1

    # Assert that the result is an EvaluationResult instance
    assert isinstance(result, EvaluationResult)

    assert result.pii_recall == pytest.approx(
        expected_recall
    ), f"Recall mismatch: expected {expected_recall}, got {result.pii_recall}"
    assert result.pii_precision == pytest.approx(
        expected_precision
    ), f"Precision mismatch: expected {expected_precision}, got {result.pii_precision}"
