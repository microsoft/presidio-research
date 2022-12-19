from typing import List, Optional

from presidio_evaluator import InputSample, Span
from presidio_evaluator.models import BaseModel


class MockModel(BaseModel):

    def predict(self, sample: InputSample) -> List[str]:
        pass

    def predict_span(self, sample: InputSample) -> List[Span]:
        pass


class MockTokensModel(BaseModel):
    """
    Simulates a real model, returns the prediction given in the constructor
    """

    def __init__(
        self,
        prediction: Optional[List[str]],
        prediction_span: Optional[List[Span]],
        entities_to_keep: List = None,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(entities_to_keep=entities_to_keep, verbose=verbose, **kwargs)
        self.prediction = prediction
        self.prediction_span = prediction_span

    def predict(self, sample: InputSample) -> List[str]:
        return self.prediction

    def predict_span(self, sample: InputSample) -> List[Span]:
        return self.prediction_span


class IdentityTokensMockModel(BaseModel):
    """
    Simulates a real model, always return the label as prediction
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)

    def predict(self, sample: InputSample) -> List[str]:
        return sample.tags

    def predict(self, sample: InputSample) -> List[Span]:
        return sample.spans


class FiftyFiftyIdentityTokensMockModel(BaseModel):
    """
    Simulates a real model, returns the label or no predictions (list of 'O')
    alternately
    """

    def __init__(self, entities_to_keep: List = None, verbose: bool = False):
        super().__init__(entities_to_keep=entities_to_keep, verbose=verbose)
        self.counter = 0

    def predict(self, sample: InputSample) -> List[str]:
        self.counter += 1
        if self.counter % 2 == 0:
            return sample.tags
        else:
            return ["O" for i in range(len(sample.tags))]


    def predict_span(self, sample: InputSample) -> List[Span]:
        return []
