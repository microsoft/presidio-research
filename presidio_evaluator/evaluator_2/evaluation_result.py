from typing import List, Dict, Tuple

import pandas as pd

from presidio_evaluator.evaluator_2 import (SampleError,
                                            evaluation_helpers)


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

    @staticmethod
    def to_span_df(span_result_dict) -> pd.DataFrame:
        """
        Convert the span_eval_schema to a pandas DataFrame
        :param span_result_dict: dictionary of span evaluation schema or span metrics
        :return: pandas DataFrame
        """
        span_eval_df = pd.DataFrame()
        for key, value in span_result_dict.items():
            temps_df = pd.DataFrame()
            for k, v in value.items():
                temps_df = temps_df.append(v, ignore_index=True)
            temps_df.insert(0, "entity", key)
            temps_df.insert(1, "eval_type", value.keys())
            span_eval_df = pd.concat([span_eval_df, temps_df], ignore_index=True)
        return span_eval_df

    def cal_span_metrics(self):
        """
        Calculate the span metrics based on the span outputs
        :param span_outputs: List of SpanOutput objects
        """
        # Step 1: convert span_outputs to a dictionary of evaluation schema
        span_outputs = []
        for sample_error in self.sample_errors:
            span_outputs += sample_error.span_output
        entities_to_keep = self.entities_to_keep

        span_eval_schema = evaluation_helpers. \
            get_span_eval_schema(span_outputs, entities_to_keep)

        # Step 2: Calculate the precision and recall for each entity type
        span_model_metrics = {}
        for key, value in span_eval_schema.items():
            span_model_metrics[key] = \
                evaluation_helpers.span_compute_precision_recall_wrapper(
                    span_eval_schema[key]
                )

        # Step 3: Calculate the f1 and fb score
        for entity, value in span_model_metrics.items():
            for k, v in value.items():
                span_model_metrics[entity][k]['f1_score'] = evaluation_helpers. \
                    span_f1_score(v['precision'], v['recall'])
                span_model_metrics[entity][k]['fb_score'] = evaluation_helpers. \
                    span_fb_score(v['precision'], v['recall'])

        # Step 4: Convert result to dataframe for display purpose
        span_eval_df = self.to_span_df(span_eval_schema)
        span_metric_df = self.to_span_df(span_model_metrics)
        span_results = span_eval_df.merge(span_metric_df, on=['entity', 'eval_type'])
        return span_eval_schema, span_model_metrics, span_results
