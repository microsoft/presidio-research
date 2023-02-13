from collections import Counter
from copy import deepcopy
from typing import List, Dict, Tuple

from presidio_evaluator.evaluator_2 import SampleError, evaluation_helpers


class EvaluationResult:
    """
    Holds the output of token and span evaluation for a given dataset
    ...

    Attributes
    ----------
    sample_errors : List[SampleError]
        contain the token, span errors and input text for further inspection
    token_confusion_matrix : Optional[Counter] = None
        list of objects of type Counter with structure {(actual, predicted) : count}
    token_model_metrics : Optional[Dict[str, Counter]] = None
        metrics calculated based on token results for the reference dataset
    span_model_metrics: Optional[Dict[str, Counter]] = None
        metrics calculated based on token results for the reference dataset
    -------
    """

    def __init__(
            self,
            sample_errors: List[SampleError] = None,
            token_confusion_matrix: Counter = None,
            token_model_metrics: Dict[str, Counter] = None
    ):
        """
        Constructs all the necessary attributes for the EvaluationResult object
        :param sample_errors: contain the token, span errors and input text for further inspection
        :param token_confusion_matrix: List of objects of type Counter
        with structure {(actual, predicted) : count}
        :param token_model_metrics: metrics calculated based on token results
        :param span_model_metrics: metrics calculated based on span results
        """

        self.sample_errors = sample_errors
        self.token_confusion_matrix = token_confusion_matrix
        self.token_model_metrics = token_model_metrics
        # Initialize span metrics
        # set up a dict for storing the span metrics
        self.span_category_output = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}
        # copy results dict to cover the four evaluation schemes for PII.
        self.span_pii_eval = {'strict': Counter(self.span_category_output),
                              'ent_type': Counter(self.span_category_output),
                              'partial': Counter(self.span_category_output),
                              'exact': Counter(self.span_category_output)}
        # copy results dict to cover the four evaluation schemes for each entity in entities_to_keep.
        self.span_model_metrics = {e: deepcopy(self.span_pii_eval) for e in self.entities_to_keep}
        self.span_model_metrics['overall'] = deepcopy(self.span_pii_eval)


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

    def get_span_eval_schema(self) -> Dict[str, Dict[str, Counter]]:
        """Update the evaluation schema with the new schema.

        param:span_outputs (dict): The new schema to update the evaluation schema with.
        returns: dict: The updated evaluation schema.
        """
        for sample_error in self.sample_errors:
            span_outputs = sample_error.span_outputs
            for span_output in span_outputs:
                entity_list = ['overall', span_output.annotated_span.entity_type]
                if span_output.output_type == "STRICT":
                    eval_list = ["strict", "ent_type", "partial", "exact"]
                    for eval_type, entity in zip(eval_list, entity_list):
                        self.span_model_metrics[entity][eval_type]["correct"] += 1
                elif span_output.output_type == "EXACT":
                    for eval_type, entity in zip(["strict", "ent_type"], entity_list):
                        self.span_model_metrics[entity][eval_type]["incorrect"] += 1
                    for eval_type, entity in zip(["partial", "exact"], entity_list):
                        self.span_model_metrics[entity][eval_type]["correct"] += 1
                elif span_output.output_type == "ENT_TYPE":
                    for entity in entity_list:
                        self.span_model_metrics[entity]["strict"]["incorrect"] += 1
                        self.span_model_metrics[entity]["ent_type"]["correct"] += 1
                        self.span_model_metrics[entity]["partial"]["partial"] += 1
                        self.span_model_metrics[entity]["exact"]["incorrect"] += 1
                elif span_output.output_type == "PARTIAL":
                    for eval_type, entity in zip(["strict", "ent_type", "exact"], entity_list):
                        self.span_model_metrics[entity][eval_type]["incorrect"] += 1
                    self.span_model_metrics["overall"]["partial"]["partial"] += 1
                    self.span_model_metrics[span_output.annotated_span.entity_type]["partial"]["partial"] += 1
                elif span_output.output_type == "SPURIOUS":
                    for eval_type, entity in zip(["strict", "ent_type", "partial", "exact"], entity_list):
                        self.span_model_metrics[entity][eval_type]["spurious"] += 1
                elif span_output.output_type == "MISSED":
                    for eval_type, entity in zip(["strict", "ent_type", "partial", "exact"], entity_list):
                        self.span_model_metrics[entity][eval_type]["missed"] += 1

    def get_possible_actual_span_pii(self):
        # Calculate the possible and actual for the whole dataset
        # at entity level
        for entity_type, entity_level in self.span_model_metrics.items():
            for eval_type in entity_level:
                self.span_model_metrics[entity_type][eval_type] = evaluation_helpers.get_actual_possible_span(
                    self.span_model_metrics[entity_type][eval_type])

    def generate_possible_actual_span_pii(self):
        # Calculate the precision and recall for the whole dataset
        # at entity level
        for entity in self.entities_to_keep:
            self.span_model_metrics[entity] = evaluation_helpers.span_compute_precision_recall_wrapper(
                self.span_model_metrics[entity])

    def get_span_model_metrics(self):
        # Generate the evaluation schema from the span outputs
        self.get_span_eval_schema()
        # Calculate the possible and actual for the whole dataset
        # at entity level
        self.get_possible_actual_span_pii()
        # Calculate the precision and recall for the whole dataset
        # at entity level
        self.generate_possible_actual_span_pii()
        print(self.span_model_metrics)
