import pytest

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import Evaluator

try:
    from flair.models import SequenceTagger
except ImportError:
    pytest.skip("Flair not available", allow_module_level=True)


from presidio_evaluator.models.flair_model import FlairModel

import numpy as np


# no-unit because flair is not a dependency by default
def test_flair_simple():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        os.path.join(dir_path, "data/generated_small.json")
    )


    flair_model = FlairModel(model_path="ner", entities_to_keep=["PERSON"])
    evaluator = Evaluator(model=flair_model)
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
