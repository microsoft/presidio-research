from presidio_evaluator import Span
from presidio_evaluator.evaluator_2 import Evaluator, SpanOutput


def test_compare_span_simple_case_1():
    annotated_spans = [
        # No matched predicted span found  - MISS case
        Span(entity_type="PER", entity_value="", start_position=59, end_position=69),
        Span(entity_type="LOC", entity_value="", start_position=127, end_position=134),
        Span(entity_type="LOC", entity_value="", start_position=164, end_position=174),
        Span(entity_type="LOC", entity_value="", start_position=197, end_position=205),
        Span(entity_type="LOC", entity_value="", start_position=208, end_position=219),
        Span(entity_type="MISC", entity_value="", start_position=230, end_position=240)]
    predicted_spans = [
        # No overlap between annotated and predicted spans - SPURIOUS case
        Span(entity_type="PER", entity_value="", start_position=24, end_position=30),
        # Entity matched and overlap ratio is < 1 - ENT_TYPE case
        Span(entity_type="LOC", entity_value="", start_position=124, end_position=134),
        # Entity doesn't match but offset is 100% overlap - EXACT case
        Span(entity_type="PER", entity_value="", start_position=164, end_position=174),
        # Entity matched and offset are 100% match - STRICT case
        Span(entity_type="LOC", entity_value="", start_position=197, end_position=205),
        # Entity matched and offset are 100% match - STRICT case
        Span(entity_type="LOC", entity_value="", start_position=208, end_position=219),
        # Entity doesn't match and overlap ratio is < 1 - PARTIAL case
        Span(entity_type="LOC", entity_value="", start_position=225, end_position=243)]

    evaluator = Evaluator(entities_to_keep=["LOC", "MISC"])
    span_outputs = evaluator.compare_span(annotated_spans, predicted_spans)
    # filter span_outputs based on entities_to_keep
    filtered_output = evaluator.filter_span_outputs_in_entities_to_keep(span_outputs)
    expected_span_outputs = [SpanOutput(output_type="ENT_TYPE",
                                        annotated_span=Span(entity_type="LOC",
                                                            entity_value="",
                                                            start_position=127,
                                                            end_position=134),
                                        predicted_span=Span(entity_type="LOC",
                                                            entity_value="",
                                                            start_position=124,
                                                            end_position=134),
                                        overlap_score=0.82),
                             SpanOutput(output_type="EXACT",
                                        annotated_span=Span(entity_type="LOC",
                                                            entity_value="",
                                                            start_position=164,
                                                            end_position=174),
                                        predicted_span=Span(entity_type="PER",
                                                            entity_value="",
                                                            start_position=164,
                                                            end_position=174),
                                        overlap_score=1),
                             SpanOutput(output_type="STRICT",
                                        annotated_span=Span(entity_type="LOC",
                                                            entity_value="",
                                                            start_position=197,
                                                            end_position=205),
                                        predicted_span=Span(entity_type="LOC",
                                                            entity_value="",
                                                            start_position=197,
                                                            end_position=205),
                                        overlap_score=1),
                             SpanOutput(output_type="STRICT",
                                        annotated_span=Span(entity_type="LOC",
                                                            entity_value="",
                                                            start_position=208,
                                                            end_position=219),
                                        predicted_span=Span(entity_type="LOC",
                                                            entity_value="",
                                                            start_position=208,
                                                            end_position=219),
                                        overlap_score=1),
                             SpanOutput(output_type="PARTIAL",
                                        annotated_span=Span(entity_type="MISC",
                                                            entity_value="",
                                                            start_position=230,
                                                            end_position=240),
                                        predicted_span=Span(entity_type="LOC",
                                                            entity_value="",
                                                            start_position=225,
                                                            end_position=243),
                                        overlap_score=0.71)]

    assert len(filtered_output) == len(expected_span_outputs)
    assert all([a.__eq__(b) for a, b in zip(filtered_output, expected_span_outputs)])