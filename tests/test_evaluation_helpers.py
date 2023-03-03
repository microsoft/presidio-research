from collections import Counter

from presidio_evaluator.data_objects import Span
from presidio_evaluator.evaluator_2 import (evaluation_helpers,
                                            SpanOutput)


def test_get_span_eval_schema():
    entities_to_keep = ["LOC", "ORG", "PER"]
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
                               annotated_span=Span(entity_type="ORG", entity_value="",
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
    # Filter the span outputs to only include the entities we want to keep
    eval_schema = evaluation_helpers. \
        get_span_eval_schema(span_outputs, entities_to_keep)
    eval_LOC = {'strict': Counter(
        {'correct': 2, 'incorrect': 2, 'partial': 0, 'missed': 0, 'spurious': 0}),
                'ent_type': Counter(
        {'correct': 3, 'incorrect': 1, 'partial': 0, 'missed': 0, 'spurious': 0}),
                'partial': Counter(
        {'correct': 3, 'partial': 1, 'incorrect': 0, 'missed': 0, 'spurious': 0}),
                'exact': Counter(
        {'correct': 3, 'incorrect': 1, 'partial': 0, 'missed': 0, 'spurious': 0})}
    eval_ORG = {'strict': Counter(
        {'incorrect': 1, 'correct': 0, 'partial': 0, 'missed': 0, 'spurious': 0}),
                'ent_type': Counter(
        {'incorrect': 1, 'correct': 0, 'partial': 0, 'missed': 0, 'spurious': 0}),
                'partial': Counter(
        {'partial': 1, 'correct': 0, 'incorrect': 0, 'missed': 0, 'spurious': 0}),
                'exact': Counter(
        {'incorrect': 1, 'correct': 0, 'partial': 0, 'missed': 0, 'spurious': 0})}
    eval_PER = {'strict': Counter(
        {'missed': 1, 'spurious': 1, 'correct': 0, 'incorrect': 0, 'partial': 0}),
                'ent_type': Counter(
        {'missed': 1, 'spurious': 1, 'correct': 0, 'incorrect': 0, 'partial': 0}),
                'partial': Counter(
        {'missed': 1, 'spurious': 1, 'correct': 0, 'incorrect': 0, 'partial': 0}),
                'exact': Counter(
        {'missed': 1, 'spurious': 1, 'correct': 0, 'incorrect': 0, 'partial': 0})}
    eval_PII = {'strict': Counter(
        {'incorrect': 3, 'correct': 2, 'missed': 1, 'spurious': 1, 'partial': 0}),
                'ent_type': Counter(
        {'correct': 3, 'incorrect': 2, 'missed': 1, 'spurious': 1, 'partial': 0}),
                'partial': Counter(
        {'correct': 3, 'partial': 2, 'missed': 1, 'spurious': 1, 'incorrect': 0}),
                'exact': Counter(
        {'correct': 3, 'incorrect': 2, 'missed': 1, 'spurious': 1, 'partial': 0})}
    assert eval_schema['LOC'] == eval_LOC
    assert eval_schema['ORG'] == eval_ORG
    assert eval_schema['PER'] == eval_PER
    assert eval_schema['PII'] == eval_PII
