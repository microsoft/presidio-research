import pytest

from presidio_evaluator import Span
from presidio_evaluator.evaluator_2 import SpanOutput, ModelPrediction


@pytest.mark.parametrize(
    "span_output1, span_output2, expected_output",
    [
        (SpanOutput(output_type="SPURIOUS",
                    predicted_span=Span(entity_type="PER", entity_value="A",
                                        start_position=24, end_position=30),
                    overlap_score=0),
         SpanOutput(output_type="SPURIOUS",
                    predicted_span=Span(entity_type="PER", entity_value="A",
                                        start_position=24, end_position=30),
                    overlap_score=0), True),
        (SpanOutput(output_type="SPURIOUS",
                    predicted_span=Span(entity_type="PER", entity_value="A",
                                        start_position=24, end_position=30),
                    overlap_score=0),
         SpanOutput(output_type="MISS",
                    annotated_span=Span(entity_type="PER", entity_value="A",
                                        start_position=24, end_position=30),
                    overlap_score=0), False)
    ],
)
def test_eq_token_output(
        span_output1, span_output2, expected_output
):
    is_eq = span_output1.__eq__(span_output2)
    assert is_eq == expected_output


@pytest.mark.parametrize(
    "predicted_tag, full_text, expected_span",
    [
        # Test 1: BIO specific input
        (['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'B-PERSON', 'I-PERSON'],
         "Someone stole my credit card. The number is 5277716201469117 and the my " \
         "name is Mary Anguiano",
         [Span(entity_type="PERSON", entity_value="Mary Anguiano", start_position=80,
               end_position=93)]),
        # Test 2: IO single input
        (['O', 'O', 'O', 'NAME'],
         "My name is Josh",
         [Span(entity_type="NAME", entity_value="Josh", start_position=11,
               end_position=15)]),
        # Test 3: BILOU multiple entities
        (['O', 'O', 'O', 'U-NAME', 'O', 'U-NAME'],
         "My name is Josh or David",
         [Span(entity_type="NAME", entity_value="Josh", start_position=11,
               end_position=15),
          Span(entity_type="NAME", entity_value="David", start_position=19,
               end_position=24)]),
        # Test 4: BIO multiple entities
        (['O', 'O', 'O', 'B-NAME', 'B-LOCATION'],
         "My name is Josh Paris",
         [Span(entity_type="NAME", entity_value="Josh", start_position=11,
               end_position=15),
          Span(entity_type="LOCATION", entity_value="Paris", start_position=16,
               end_position=21)]),
        # Test 5: BIO single entity
        (['O', 'O', 'O', 'O', 'O', 'B-PERSON', 'L-PERSON', 'O', 'O', 'O'],
         "May I get access to Jessica Gump's account?",
         [Span(entity_type="PERSON", entity_value="Jessica Gump", start_position=20,
               end_position=32)]),
        # Test 6: BIO single entity
        (['O', 'O', 'O', 'B-ADDRESS', 'I-ADDRESS', 'I-ADDRESS',
          'I-ADDRESS', 'I-ADDRESS', 'I-ADDRESS', 'O', 'O', 'O', 'O', 'O'],
         "My Address is 409 Bob st. Manhattan NY. I just moved in",
         [Span(entity_type="ADDRESS", entity_value="409 Bob st. Manhattan NY",
               start_position=14, end_position=38)]),
    ],
)
def test_tag_to_span(predicted_tag, full_text, expected_span):
    model_prediction = ModelPrediction(predicted_tag)
    predicted_span = model_prediction.tag_to_span(predicted_tag, full_text)
    assert predicted_span.__eq__(expected_span)
