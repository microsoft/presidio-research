import pandas as pd
import pytest
from presidio_evaluator.evaluation.span_evaluator import SpanEvaluator
from presidio_evaluator.data_objects import Span

@pytest.fixture
def span_evaluator():
    return SpanEvaluator(iou_threshold=0.5)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'sentence_id': [0, 0, 0, 0, 0, 0, 0],
        'token': ['John', 'Smith', 'lives', 'in', 'New', 'York', '.'],
        'start': [0, 5, 11, 17, 20, 24, 28],
        'annotation': ['PERSON', 'PERSON', 'O', 'O', 'LOCATION', 'LOCATION', 'O'],
        'prediction': ['PERSON', 'PERSON', 'O', 'O', 'LOCATION', 'LOCATION', 'O']
    })

@pytest.fixture
def complex_df():
    return pd.DataFrame({
        'sentence_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'token': ['The', 'United', 'States', 'of', 'America', 'and', 'New', 'York', 'City', '.'],
        'start': [0, 4, 11, 18, 21, 29, 33, 37, 42, 46],
        'annotation': ['O', 'LOCATION', 'LOCATION', 'LOCATION', 'LOCATION', 'O', 'LOCATION', 'LOCATION', 'LOCATION', 'O'],
        'prediction': ['O', 'LOCATION', 'LOCATION', 'O', 'LOCATION', 'O', 'LOCATION', 'LOCATION', 'O', 'O']
    })

def test_normalize_tokens(span_evaluator):
    tokens = ['The', 'United', 'States', 'of', 'America', '.']
    normalized = span_evaluator.normalize_tokens(tokens)
    assert normalized == ['united', 'states', 'america']

def test_merge_adjacent_spans(span_evaluator, sample_df):
    spans = [
        Span('PERSON', ['John'], 0, 5),
        Span('PERSON', ['Smith'], 5, 10),
        Span('LOCATION', ['New'], 20, 24),
        Span('LOCATION', ['York'], 24, 28)
    ]
    
    merged = span_evaluator.merge_adjacent_spans(spans, sample_df)
    assert len(merged) == 2
    assert merged[0].start_position == 0
    assert merged[0].end_position == 10  # End of "Smith"
    assert merged[1].start_position == 20  # Start of "New"
    assert merged[1].end_position == 28  # End of "York"

def test_calculate_iou(span_evaluator, sample_df):
    span1 = Span('PERSON', ['John', 'Smith'], 0, 10)
    span2 = Span('PERSON', ['John', 'Smith'], 0, 10)
    iou = span_evaluator.calculate_iou(span1, span2, sample_df)
    assert iou == 1.0

    span3 = Span('PERSON', ['John'], 0, 5)
    iou = span_evaluator.calculate_iou(span1, span3, sample_df)
    assert iou == 0.5

def test_evaluate_perfect_match(span_evaluator, sample_df):
    results = span_evaluator.evaluate(sample_df)
    assert results['precision'] == 1.0
    assert results['recall'] == 1.0
    assert results['f1'] == 1.0
    
    # Check per-type metrics
    assert results['per_type']['PERSON']['precision'] == 1.0
    assert results['per_type']['LOCATION']['precision'] == 1.0

def test_evaluate_partial_match(span_evaluator, complex_df):
    results = span_evaluator.evaluate(complex_df)
    
    # Should find partial matches for both locations
    assert results['precision'] > 0.0
    assert results['recall'] > 0.0
    assert results['f1'] > 0.0
    

def test_evaluate_empty_prediction(span_evaluator):
    df = pd.DataFrame({
        'sentence_id': [0, 0, 0],
        'token': ['John', 'Smith', '.'],
        'start': [0, 5, 10],
        'annotation': ['PERSON', 'PERSON', 'O'],
        'prediction': ['O', 'O', 'O']
    })
    
    results = span_evaluator.evaluate(df)
    assert results['precision'] == 0.0
    assert results['recall'] == 0.0
    assert results['f1'] == 0.0

def test_evaluate_no_annotation(span_evaluator):
    df = pd.DataFrame({
        'sentence_id': [0, 0, 0],
        'token': ['John', 'Smith', '.'],
        'start': [0, 5, 10],
        'annotation': ['O', 'O', 'O'],
        'prediction': ['PERSON', 'PERSON', 'O']
    })
    
    results = span_evaluator.evaluate(df)
    assert results['precision'] == 0.0
    assert results['recall'] == 0.0
    assert results['f1'] == 0.0

