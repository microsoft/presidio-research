from collections import Counter

import pytest

from presidio_evaluator.evaluation import EvaluationResult, Evaluator
from tests.mocks import (
    MockTokensModel,
)


@pytest.fixture(scope="session")
def scores():
    results = Counter(
        {
            ("O", "O"): 30,
            ("ANIMAL", "ANIMAL"): 4,
            ("ANIMAL", "O"): 2,
            ("O", "ANIMAL"): 1,
            ("PERSON", "PERSON"): 2,
        }
    )
    model = MockTokensModel(prediction=None)
    evaluator = Evaluator(model=model)
    evaluation_result = EvaluationResult(results=results)

    return evaluator.calculate_score([evaluation_result])


def test_to_confusion_matrix(scores):
    entities, confmatrix = scores.to_confusion_matrix()
    assert "O" in entities
    assert "PERSON" in entities
    assert "ANIMAL" in entities
    assert confmatrix == [[4, 2, 0], [1, 30, 0], [0, 0, 2]]


def test_str(scores):
    return_str = str(scores)
    assert (
        "PERSON             100.00%             100.00%                   2"
        in return_str
    )
    assert (
        "ANIMAL              80.00%              66.67%                   6"
        in return_str
    )
    assert (
        "PII              85.71%              75.00%                   8" in return_str
    )
