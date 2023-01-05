import json
from collections import Counter
from typing import List, Optional, Dict, Tuple

from presidio_evaluator.evaluation import SampleError


class EvaluationResult:
    def __init__(
        self,
        sample_errors: List[SampleError],
        token_confusion_matrix: Counter,
        token_model_metrics: Dict[str, Dict[str, float]],
        span_model_metrics: Dict[str, Dict[str, float]]

    ):
        """
        Holds the output of token and span evaluation for a given dataset
        :param model_errors: List of token and span errors for further inspection
        :param token_confusion_matrix: List of objects of type Counter
        with structure {(actual, predicted) : count}
        :param token_model_metrics: metrics calculated based on token results
        :param span_model_metrics: metrics calculated based on span results
        """

        self.sample_errors = sample_errors
        self.token_confusion_matrix = token_confusion_matrix
        self.token_model_metrics = token_model_metrics
        self.span_model_metrics = span_model_metrics

    # TODO: Refactor those functions
    # def __str__(self):
    #     return_str = ""
    #     if not self.entity_precision_dict or not self.entity_recall_dict:
    #         return json.dumps(self.results)

    #     entities = self.n_dict.keys()

    #     row_format = "{:>20}{:>20.2%}{:>20.2%}{:>20}"
    #     header_format = "{:>20}" * 4
    #     return_str += str(
    #         header_format.format(
    #             *("Entity", "Precision", "Recall", "Number of samples")
    #         )
    #     )
    #     for entity in entities:
    #         return_str += "\n" + row_format.format(
    #             entity,
    #             self.entity_precision_dict[entity],
    #             self.entity_recall_dict[entity],
    #             self.n_dict[entity],
    #         )

    #     # add PII values
    #     return_str += "\n" + row_format.format(
    #         "PII",
    #         self.pii_precision,
    #         self.pii_recall,
    #         self.n,
    #     )

    #     return_str += f"\nPII F measure: {self.pii_f:.2%}"
    #     return return_str

    # def __repr__(self):
    #     return f"stats={self.results}"

    # def to_log(self):
    #     metrics_dict = {
    #         "pii_f": self.pii_f,
    #     }
    #     if self.entity_precision_dict:
    #         metrics_dict.update(
    #             {
    #                 f"{ent}_precision": v
    #                 for (ent, v) in self.entity_precision_dict.items()
    #             }
    #         )
    #     if self.entity_recall_dict:
    #         metrics_dict.update(
    #             {f"{ent}_recall": v for (ent, v) in self.entity_recall_dict.items()}
    #         )
    #     if self.n:
    #         metrics_dict.update(self.n_dict)
    #     return metrics_dict

    # def to_confusion_matrix(self) -> Tuple[List[str], List[List[int]]]:
    #     entities = sorted(list(set(self.n_dict.keys()).union("O")))
    #     confusion_matrix = [[0] * len(entities) for _ in range(len(entities))]
    #     for i, actual in enumerate(entities):
    #         for j, predicted in enumerate(entities):
    #             confusion_matrix[i][j] = self.results[(actual, predicted)]

    #     return entities, confusion_matrix
