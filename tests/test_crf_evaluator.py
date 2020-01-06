import numpy as np

from presidio_evaluator.crf_evaluator import CRFEvaluator
from presidio_evaluator.data_generator import read_synth_dataset


# no_test since the CRF model is not supplied with the package
def no_test_test_crf_simple():
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = read_synth_dataset(os.path.join(dir_path, "data/generated_small.txt"))

    model_path = os.path.abspath(os.path.join(dir_path, "..", "model-outputs/crf.pickle"))

    crf_evaluator = CRFEvaluator(model_pickle_path=model_path,entities_to_keep=['PERSON'])
    evaluation_results = crf_evaluator.evaluate_all(input_samples)
    scores = crf_evaluator.calculate_score(evaluation_results)

    np.testing.assert_almost_equal(scores.pii_precision, scores.entity_precision_dict['PERSON'])
    np.testing.assert_almost_equal(scores.pii_recall, scores.entity_recall_dict['PERSON'])
    assert scores.pii_recall > 0
    assert scores.pii_precision > 0
