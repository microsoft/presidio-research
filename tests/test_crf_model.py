import os
import sys

import pytest

from presidio_evaluator import split_dataset
from presidio_evaluator.evaluation import Evaluator
from presidio_evaluator.models.crf_model import CRFModel
from tests.conftest import assert_model_results_gt

try:
    import sklearn_crfsuite
except ImportError:
    sklearn_crfsuite = None


@pytest.mark.skipif(
    sklearn_crfsuite is None, reason="requires the sklearn_crfsuite library"
)
def test_crf_simple(small_dataset):
    train_test_ratios = [0.7, 0.3]

    train, test = split_dataset(small_dataset, train_test_ratios)

    crf_model = CRFModel(model_pickle_path=None, entities_to_keep=["PERSON"])
    crf_model.fit(train)
    evaluator = Evaluator(model=crf_model)
    evaluation_results = evaluator.evaluate_all(test)
    scores = evaluator.calculate_score(evaluation_results)

    assert_model_results_gt(scores, "PERSON", 0)
