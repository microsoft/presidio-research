from typing import List, Optional

from presidio_evaluator import InputSample
from presidio_evaluator.models_2 import BaseModel
from presidio_evaluator.evaluator_2 import ModelPrediction


class MockModel(BaseModel):

    def predict(self, sample: InputSample) -> ModelPrediction:
        pass


class MockTokensModel(BaseModel):
    """
    Simulates a real model, returns the prediction given in the constructor
    """

    def __init__(
        self,
        prediction: Optional[List[str]],
        entities_to_keep: List = None,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(entities_to_keep=entities_to_keep, verbose=verbose, **kwargs)
        self.prediction = prediction

    def predict(self, sample: InputSample) -> ModelPrediction:
        return ModelPrediction(
            input_sample=sample,
            predicted_tags=self.prediction)


class IdentityTokensMockModel(BaseModel):
    """
    Simulates a real model, always return the label as prediction
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)

    def predict(self, sample: InputSample) -> ModelPrediction:
        return ModelPrediction(
            input_sample=sample,
            predicted_tags=sample.tags)


class FiftyFiftyIdentityTokensMockModel(BaseModel):
    """
    Simulates a real model, returns the label or no predictions (list of 'O')
    alternately
    """

    def __init__(self, entities_to_keep: List = None, verbose: bool = False):
        super().__init__(entities_to_keep=entities_to_keep, verbose=verbose)
        self.counter = 0

    def predict(self, sample: InputSample) -> ModelPrediction:
        self.counter += 1
        if self.counter % 2 == 0:
            return ModelPrediction(
                input_sample=sample,
                predicted_tags=sample.tags)
        else:
            return ModelPrediction(
                input_sample=sample,
                predicted_tags=["O" for i in range(len(sample.tags))])
