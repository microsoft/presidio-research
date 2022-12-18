from typing import List, Dict
import pandas as pd

from presidio_evaluator.evaluation import SpanOutput


class SpanEvaluationResult:
    def __init__(
        self,
        span_outputs: List[SpanOutput],
        model_metrics: Dict[str, Dict],
    ):
        """
        Hold the ouput of the comparison between ground truth and predicted spans
        :param model_errors: List of spans output in SpanOutput format
        :param model_metrics: Dictionary contains the strict and flexiable precision/recall
        """
        self.span_outputs = span_outputs
        self.model_metrics = model_metrics

    
    def visualize_metric(self):
        """
        Visualize the span output and metrics of the evaluation at the span level in dataframe format
        """
        df_span_output = pd.DataFrame(self.model_metrics["span_output"])
        df_metrics = pd.DataFrame(self.model_metrics["metrics"])
        # Remove the entity that contains only zeros
        df_span_output = df_span_output.loc[:, (df_span_output != 0).any(axis=0)]
        df_metrics = df_metrics.loc[:, (df_metrics != 0).any(axis=0)]
        return df_span_output, df_metrics
