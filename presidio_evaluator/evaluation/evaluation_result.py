from collections import Counter
from typing import List, Optional
from presidio_evaluator.evaluation import ModelError


class EvaluationResult(object):
    def __init__(
        self,
        results: Counter,
        model_errors: Optional[List[ModelError]] = None,
        text: str = None,
    ):
        """
        Holds the output of a comparison between ground truth and predicted
        :param results: List of objects of type Counter
        with structure {(actual, predicted) : count}
        :param model_errors: List of specific model errors for further inspection
        :param text: sample's full text (if used for one sample)
        """

        self.results = results
        self.model_errors = model_errors
        self.text = text

        self.pii_recall = None
        self.pii_precision = None
        self.pii_f = None
        self.entity_recall_dict = None
        self.entity_precision_dict = None

    def print(self):
        recall_dict = self.entity_recall_dict
        precision_dict = self.entity_precision_dict

        recall_dict["PII"] = self.pii_recall
        precision_dict["PII"] = self.pii_precision

        entities = recall_dict.keys()
        recall = recall_dict.values()
        precision = precision_dict.values()

        row_format = "{:>30}{:>30.2%}{:>30.2%}"
        header_format = "{:>30}" * 3
        print(header_format.format(*("Entity", "Precision", "Recall")))
        for entity, precision, recall in zip(entities, precision, recall):
            print(row_format.format(entity, precision, recall))

        print("PII F measure: {}".format(self.pii_f))

    def __repr__(self):
        return f"stats={self.results}"
