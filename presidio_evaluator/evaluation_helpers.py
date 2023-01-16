import numpy as np
from typing import List, Dict
from collections import Counter

from presidio_evaluator import Span
from presidio_evaluator.evaluation import SpanOutput


def get_matched_gold(predicted_span: Span, 
                    annotated_span: List[Span], 
                    overlap_threshold) -> SpanOutput:
        """
        Given a predicted_span, get the best matchest annotated_span based on the overlap_threshold. 
        Return a SpanOutput
        :param sample: InputSample
        :param pred_span: Span,  Predicted span
        :param gold_span: List[Span]: List of gold spans from the annotation input
        """
        return SpanOutput(output_type="",
                            predicted_span=None,
                            annotated_span=None,
                            overlap_score=0
                            )

def span_compute_actual_possible(results: dict) -> dict:
        """
        Take the result dict and calculate the actual and possible spans
        """
        strict = results["strict"]
        exact = results["exact"]
        incorrect = results["incorrect"]
        partial = results["partial"]
        missed = results["miss"]
        spurious = results["spurious"]
        # Possible: Number of annotations in the gold-standard which contribute to the final score
        possible = strict + exact + incorrect + partial + missed
        # Actual: Number of annotations produced by the PII detection system
        actual = strict + exact + incorrect + partial + spurious

        results["actual"] = actual
        results["possible"] = possible
        
        return results

def span_compute_precision_recall(results: dict) -> dict:
    """
    Take the result dict to calculate the strict and flexible precision/ recall
    """
    metrics = {}
    strict = results["strict"]
    exact = results["exact"]
    partial = results["partial"]
    actual = results["actual"]
    possible = results["possible"]
    
    # Calculate the strict performance
    strict_precision = strict / actual if actual > 0 else 0
    strict_recall = strict / possible if possible > 0 else 0

    # Calculate the flexible performance
    flexible_precision = (strict + exact)/ actual if actual > 0 else 0
    flexible_recall = (strict + exact) / possible if possible > 0 else 0

    # Calculate the partial performance
    partial_precision = (strict + exact + 0.5 * partial) / actual if actual > 0 else 0
    partial_recall = (strict + exact + 0.5 * partial) / possible if possible > 0 else 0
    

    metrics["strict precision"] = strict_precision
    metrics["strict recall"] = strict_recall
    metrics["flexible precision"] = flexible_precision
    metrics["flexible recall"] = flexible_recall
    metrics["partial precision"] = partial_precision
    metrics["partial recall"] = partial_recall
    return metrics

# TODO: Implement this function
def dict_merge(dict_1: dict, dict2: dict) -> dict:
    """
    Examples: Sum up the value of two dictionaries by keys 
    >>> dict_1 = {'PII': {
                        'correct': 2,
                        'partial': 1
                    },
                    'PERSON': {
                        'correct': 2,
                        'partial': 0,
                    }
                }
    >>> dict_2 = {'PII': {
                        'correct': 3,
                        'partial': 0
                    },
                    'PERSON': {
                        'correct': 1,
                        'partial': 1,
                    }
                }    
    >>> dict_merge(dict1, dict2)
    {'PII': {
                'correct': 5,
                'partial': 1
            },
    'PERSON': {
        'correct': 3,
        'partial': 1,
            }
    }
    """
    results = {}
    return results

# TODO: Implement this function
def token_calulate_score(token_confusion_matrix: Counter) -> Dict:
    """
    Calculate the token model metrics from token confusion matrix
    Examples: Sum up the value of two dictionaries by keys 
    >>> token_confusion_matrix = Counter({('O', 'O'): X, ('O', 'DateTime'): X, ('DateTime', 'O'): X, ('DateTime', 'DateTime'): X})
    >>> token_calulate_score(token_confusion_matrix)
    {'PII': {
                'recall': xxx,
                'precision': xxx,
                'F measure': xxx
            },
    'PERSON': {
                'recall': xxx,
                'precision': xxx,
    }
    }
    """
    token_model_metrics = {}
    return token_model_metrics

    