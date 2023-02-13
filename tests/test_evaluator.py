from collections import Counter

from presidio_evaluator import Span
from presidio_evaluator.evaluator_2 import Evaluator, SpanOutput


def test_compare_span_simple_case_1():
    annotated_spans = [Span(entity_type="PER", entity_value="", start_position=59, end_position=69),
                       Span(entity_type="LOC", entity_value="", start_position=127, end_position=134),
                       Span(entity_type="LOC", entity_value="", start_position=164, end_position=174),
                       Span(entity_type="LOC", entity_value="", start_position=197, end_position=205),
                       Span(entity_type="LOC", entity_value="", start_position=208, end_position=219),
                       Span(entity_type="MISC", entity_value="", start_position=230, end_position=240)]
    predicted_spans = [Span(entity_type="PER", entity_value="", start_position=24, end_position=30),
                       Span(entity_type="LOC", entity_value="", start_position=124, end_position=134),
                       Span(entity_type="PER", entity_value="", start_position=164, end_position=174),
                       Span(entity_type="LOC", entity_value="", start_position=197, end_position=205),
                       Span(entity_type="LOC", entity_value="", start_position=208, end_position=219),
                       Span(entity_type="LOC", entity_value="", start_position=225, end_position=243)]

    evaluator = Evaluator(entities_to_keep=['PER', 'LOC', 'MISC'])
    span_outputs = evaluator.compare_span(annotated_spans, predicted_spans)
    expected_span_outputs = [SpanOutput(output_type="SPURIOUS",
                                        predicted_span=Span(entity_type="PER", entity_value="", start_position=24,
                                                            end_position=30),
                                        overlap_score=0),
                             SpanOutput(output_type="ENT_TYPE",
                                        annotated_span=Span(entity_type="LOC", entity_value="", start_position=127,
                                                            end_position=134),
                                        predicted_span=Span(entity_type="LOC", entity_value="", start_position=124,
                                                            end_position=134),
                                        overlap_score=0.82),
                             SpanOutput(output_type="EXACT",
                                        annotated_span=Span(entity_type="LOC", entity_value="", start_position=164,
                                                            end_position=174),
                                        predicted_span=Span(entity_type="PER", entity_value="", start_position=164,
                                                            end_position=174),
                                        overlap_score=1),
                             SpanOutput(output_type="STRICT",
                                        annotated_span=Span(entity_type="LOC", entity_value="", start_position=197,
                                                            end_position=205),
                                        predicted_span=Span(entity_type="LOC", entity_value="", start_position=197,
                                                            end_position=205),
                                        overlap_score=1),
                             SpanOutput(output_type="STRICT",
                                        annotated_span=Span(entity_type="LOC", entity_value="", start_position=208,
                                                            end_position=219),
                                        predicted_span=Span(entity_type="LOC", entity_value="", start_position=208,
                                                            end_position=219),
                                        overlap_score=1),
                             SpanOutput(output_type="PARTIAL",
                                        annotated_span=Span(entity_type="MISC", entity_value="", start_position=230,
                                                            end_position=240),
                                        predicted_span=Span(entity_type="LOC", entity_value="", start_position=225,
                                                            end_position=243),
                                        overlap_score=0.71),
                             SpanOutput(output_type="MISSED",
                                        annotated_span=Span(entity_type="PER", entity_value="", start_position=59,
                                                            end_position=69), overlap_score=0)]

    assert len(span_outputs) == len(expected_span_outputs)
    assert all([a.__eq__(b) for a, b in zip(span_outputs, expected_span_outputs)])


def test_get_span_eval_schema():
    evaluator = Evaluator(entities_to_keep=['PER', 'LOC', 'MISC'])
    span_outputs = [SpanOutput(output_type="SPURIOUS",
                               predicted_span=Span(entity_type="PER", entity_value="", start_position=24,
                                                   end_position=30),
                               overlap_score=0),
                    SpanOutput(output_type="ENT_TYPE",
                               annotated_span=Span(entity_type="LOC", entity_value="", start_position=127,
                                                   end_position=134),
                               predicted_span=Span(entity_type="LOC", entity_value="", start_position=124,
                                                   end_position=134),
                               overlap_score=0.82),
                    SpanOutput(output_type="EXACT",
                               annotated_span=Span(entity_type="LOC", entity_value="", start_position=164,
                                                   end_position=174),
                               predicted_span=Span(entity_type="PER", entity_value="", start_position=164,
                                                   end_position=174),
                               overlap_score=1),
                    SpanOutput(output_type="STRICT",
                               annotated_span=Span(entity_type="LOC", entity_value="", start_position=197,
                                                   end_position=205),
                               predicted_span=Span(entity_type="LOC", entity_value="", start_position=197,
                                                   end_position=205),
                               overlap_score=1),
                    SpanOutput(output_type="STRICT",
                               annotated_span=Span(entity_type="LOC", entity_value="", start_position=208,
                                                   end_position=219),
                               predicted_span=Span(entity_type="LOC", entity_value="", start_position=208,
                                                   end_position=219),
                               overlap_score=1),
                    SpanOutput(output_type="PARTIAL",
                               annotated_span=Span(entity_type="MISC", entity_value="", start_position=230,
                                                   end_position=240),
                               predicted_span=Span(entity_type="LOC", entity_value="", start_position=225,
                                                   end_position=243),
                               overlap_score=0.71),
                    SpanOutput(output_type="MISSED",
                               annotated_span=Span(entity_type="PER", entity_value="", start_position=59,
                                                   end_position=69), overlap_score=0)]
    evaluator.get_span_eval_schema(span_outputs=span_outputs)
    print(evaluator.span_pii_eval)
    expected_schema = {'strict': Counter({'incorrect': 3, 'correct': 2, 'missed': 1, 'spurious': 1, 'partial': 0}),
                       'ent_type': Counter({'correct': 3, 'incorrect': 2, 'missed': 1, 'spurious': 1, 'partial': 0}),
                       'partial': Counter({'correct': 3, 'partial': 2, 'missed': 1, 'spurious': 1, 'incorrect': 0}),
                       'exact': Counter({'correct': 3, 'incorrect': 2, 'missed': 1, 'spurious': 1, 'partial': 0})}
    assert evaluator.span_pii_eval == expected_schema
