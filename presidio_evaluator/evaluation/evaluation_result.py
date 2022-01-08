import json
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
        self.n = None

    def print(self):
        if not self.entity_precision_dict or not self.entity_recall_dict:
            return json.dumps(self.results)

        recall_dict = dict(sorted(self.entity_recall_dict.items()))
        precision_dict = dict(sorted(self.entity_precision_dict.items()))

        recall_dict["PII"] = self.pii_recall
        precision_dict["PII"] = self.pii_precision

        entities = recall_dict.keys()
        recall = recall_dict.values()
        precision = precision_dict.values()
        n = self.n.values()

        row_format = "{:>30}{:>30.2%}{:>30.2%}{:>30}"
        header_format = "{:>30}" * 4
        print(
            header_format.format(
                *("Entity", "Precision", "Recall", "Number of samples")
            )
        )
        for entity, precision, recall, n in zip(entities, precision, recall, n):
            print(row_format.format(entity, precision, recall, n))

        print("PII F measure: {}".format(self.pii_f))

    def __repr__(self):
        return f"stats={self.results}"

    def to_log(self):
        metrics_dict = {
            "pii_f": self.pii_f,
        }
        if self.entity_precision_dict:
            metrics_dict.update(
                {
                    f"{ent}_precision": v
                    for (ent, v) in self.entity_precision_dict.items()
                }
            )
        if self.entity_recall_dict:
            metrics_dict.update(
                {
                    f"{ent}_recall": v
                    for (ent, v) in self.entity_recall_dict.items()
                }
            )
        if self.n:
            metrics_dict.update(self.n)
        return metrics_dict
