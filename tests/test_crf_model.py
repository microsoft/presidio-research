import numpy as np
import pytest

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import Evaluator
from presidio_evaluator.models.crf_model import CRFModel


# no_test since the CRF model is not supplied with the package
@pytest.mark.skip(reason="CRF suite is not installed by default")
def test_test_crf_simple():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        os.path.join(dir_path, "data/generated_small.json")
    )

    model_path = os.path.abspath(
        os.path.join(dir_path, "..", "model-outputs/crf.pickle")
    )

    crf_model = CRFModel(model_pickle_path=model_path, entities_to_keep=["PERSON"])
    evaluator = Evaluator(model=crf_model)
    evaluation_results = evaluator.evaluate_all(input_samples)
    scores = evaluator.calculate_score(evaluation_results)

    np.testing.assert_almost_equal(
        scores.pii_precision, scores.entity_precision_dict["PERSON"]
    )
    np.testing.assert_almost_equal(
        scores.pii_recall, scores.entity_recall_dict["PERSON"]
    )
    assert scores.pii_recall > 0
    assert scores.pii_precision > 0
