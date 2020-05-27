import pytest

try:
    from flair.models import SequenceTagger
except:
    ImportError("Flair is not installed by default")

from presidio_evaluator.data_generator import read_synth_dataset
from presidio_evaluator.flair_evaluator import FlairEvaluator

import numpy as np

# no-unit because flair is not a dependency by default
@pytest.mark.skip(reason="Flair not installed by default")
def test_flair_simple():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = read_synth_dataset(
        os.path.join(dir_path, "data/generated_small.txt")
    )

    model = SequenceTagger.load("ner-ontonotes-fast")  # .load('ner')

    flair_evaluator = FlairEvaluator(model=model, entities_to_keep=["PERSON"])
    evaluation_results = flair_evaluator.evaluate_all(input_samples)
    scores = flair_evaluator.calculate_score(evaluation_results)

    np.testing.assert_almost_equal(
        scores.pii_precision, scores.entity_precision_dict["PERSON"]
    )
    np.testing.assert_almost_equal(
        scores.pii_recall, scores.entity_recall_dict["PERSON"]
    )
    assert scores.pii_recall > 0
    assert scores.pii_precision > 0
