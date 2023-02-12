import pytest

from presidio_evaluator import Span
from presidio_evaluator.evaluator_2 import SpanOutput


@pytest.mark.parametrize(
    "span_output1, span_output2, expected_output",
    [
        (SpanOutput(output_type="SPURIOUS",
                    predicted_span=Span(entity_type="PER", entity_value="A", start_position=24, end_position=30),
                    overlap_score=0),
         SpanOutput(output_type="SPURIOUS",
                    predicted_span=Span(entity_type="PER", entity_value="A", start_position=24, end_position=30),
                    overlap_score=0), True),
        (SpanOutput(output_type="SPURIOUS",
                    predicted_span=Span(entity_type="PER", entity_value="A", start_position=24, end_position=30),
                    overlap_score=0),
         SpanOutput(output_type="MISS",
                    annotated_span=Span(entity_type="PER", entity_value="A", start_position=24, end_position=30),
                    overlap_score=0), False)
    ],
)
def test_eq_token_output(
        span_output1, span_output2, expected_output
):
    is_eq = span_output1.__eq__(span_output2)
    assert is_eq == expected_output
