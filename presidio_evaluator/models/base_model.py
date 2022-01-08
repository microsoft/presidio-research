from abc import ABC, abstractmethod
from typing import List, Dict

from presidio_evaluator import InputSample


class BaseModel(ABC):
    def __init__(
        self,
        labeling_scheme: str = "BILUO",
        entities_to_keep: List[str] = None,
        verbose: bool = False,
    ):

        """
        Abstract class for evaluating NER models and others
        :param entities_to_keep: Which entities should be evaluated? All other
        entities are ignored. If None, none are filtered
        :param labeling_scheme: Used to translate (if needed)
        the prediction to a specific scheme (IO, BIO/IOB, BILUO)
        :param verbose: Whether to print more debug info


        """
        self.entities = entities_to_keep
        self.labeling_scheme = labeling_scheme
        self.verbose = verbose

    @abstractmethod
    def predict(self, sample: InputSample) -> List[str]:
        """
        Abstract. Returns the predicted tokens/spans from the evaluated model
        :param sample: Sample to be evaluated
        :return: if self.use spans: list of spans
                 if not self.use_spans: tags in self.labeling_scheme format
        """
        pass

    def to_log(self) -> Dict:
        """
        Returns a dictionary of parameters for logging purposes.
        :return:
        """
        return {
            "labeling_scheme": self.labeling_scheme,
            "entities_to_keep": self.entities,
        }
