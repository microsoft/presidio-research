import pandas as pd

from presidio_evaluator.data_objects import Span
from presidio_evaluator.evaluator_2 import (SpanOutput,
                                            SampleError,
                                            EvaluationResult)


def test_cal_span_metrics():
    span_outputs = [SpanOutput(output_type="SPURIOUS",
                               predicted_span=Span(entity_type="PER", entity_value="",
                                                   start_position=24,
                                                   end_position=30),
                               overlap_score=0),
                    SpanOutput(output_type="ENT_TYPE",
                               annotated_span=Span(entity_type="LOC", entity_value="",
                                                   start_position=127,
                                                   end_position=134),
                               predicted_span=Span(entity_type="LOC", entity_value="",
                                                   start_position=124,
                                                   end_position=134),
                               overlap_score=0.82),
                    SpanOutput(output_type="EXACT",
                               annotated_span=Span(entity_type="LOC", entity_value="",
                                                   start_position=164,
                                                   end_position=174),
                               predicted_span=Span(entity_type="PER", entity_value="",
                                                   start_position=164,
                                                   end_position=174),
                               overlap_score=1),
                    SpanOutput(output_type="SPURIOUS",
                               predicted_span=Span(entity_type="LOC", entity_value="",
                                                   start_position=40,
                                                   end_position=50),
                               overlap_score=0),
                    SpanOutput(output_type="STRICT",
                               annotated_span=Span(entity_type="LOC", entity_value="",
                                                   start_position=197,
                                                   end_position=205),
                               predicted_span=Span(entity_type="LOC", entity_value="",
                                                   start_position=197,
                                                   end_position=205),
                               overlap_score=1),
                    SpanOutput(output_type="STRICT",
                               annotated_span=Span(entity_type="LOC", entity_value="",
                                                   start_position=208,
                                                   end_position=219),
                               predicted_span=Span(entity_type="LOC", entity_value="",
                                                   start_position=208,
                                                   end_position=219),
                               overlap_score=1),
                    SpanOutput(output_type="PARTIAL",
                               annotated_span=Span(entity_type="MISC", entity_value="",
                                                   start_position=230,
                                                   end_position=240),
                               predicted_span=Span(entity_type="LOC", entity_value="",
                                                   start_position=225,
                                                   end_position=243),
                               overlap_score=0.71),
                    SpanOutput(output_type="MISSED",
                               annotated_span=Span(entity_type="PER", entity_value="",
                                                   start_position=59,
                                                   end_position=69), overlap_score=0)]
    sample_errors = [SampleError(
        span_output=span_outputs
    )]
    evaluation_result = EvaluationResult(sample_errors=sample_errors,
                                         entities_to_keep=["LOC", "PER", "MISC"])
    span_eval_schema, span_metrics_dict = evaluation_result.cal_span_metrics()

    span_eval_df = evaluation_result.to_span_df(span_eval_schema)
    span_metric_df = evaluation_result.to_span_df(span_metrics_dict)
    print(span_eval_df)
    print(span_metric_df)
    assert 0 == 1
