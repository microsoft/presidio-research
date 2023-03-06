from typing import List, Dict, Tuple

from presidio_evaluator.evaluator_2 import SampleError


class EvaluationResult:
    def __init__(
            self,
            sample_errors: List[SampleError] = None,
            entities_to_keep: List[str] = None,
    ):
        """
        Constructs all the necessary attributes for the EvaluationResult object
        :param sample_errors: contain the token, span errors and input text
        for further inspection
        :param entities_to_keep: List of entity names to focus the evaluator on
        """

        self.sample_errors = sample_errors
        self.entities_to_keep = entities_to_keep

    def to_log(self) -> Dict:
        """
        Reformat the EvaluationResult to log the output
        """
        pass

    def to_confusion_matrix(self) -> Tuple[List[str], List[List[int]]]:
        """
        Convert the EvaluationResult to display confusion matrix to the end user
        """
        pass
