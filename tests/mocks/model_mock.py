from typing import List

from presidio_evaluator import InputSample, ModelEvaluator


class MockTokensModel(ModelEvaluator):
    """
    Simulates a real model, returns the prediction given in the constructor
    """

    def __init__(self, prediction: List[str], entities_to_keep: List = None,
                 verbose: bool = False, **kwargs):
        super().__init__(entities_to_keep=entities_to_keep, verbose=verbose,
                         **kwargs)
        self.prediction = prediction

    def predict(self, sample: InputSample) -> List[str]:
        return self.prediction


class IdentityTokensMockModel(ModelEvaluator):
    """
    Simulates a real model, always return the label as prediction
    """

    def __init__(self, entities_to_keep: List = None,
                 verbose: bool = False):
        super().__init__(entities_to_keep=entities_to_keep, verbose=verbose)

    def predict(self, sample: InputSample) -> List[str]:
        return sample.tags


class FiftyFiftyIdentityTokensMockModel(ModelEvaluator):
    """
    Simulates a real model, returns the label or no predictions (list of 'O')
    alternately
    """

    def __init__(self, entities_to_keep: List = None,
                 verbose: bool = False):
        super().__init__(entities_to_keep=entities_to_keep, verbose=verbose)
        self.counter = 0

    def predict(self, sample: InputSample) -> List[str]:
        self.counter += 1
        if self.counter % 2 == 0:
            return sample.tags
        else:
            return ["O" for i in range(len(sample.tags))]
