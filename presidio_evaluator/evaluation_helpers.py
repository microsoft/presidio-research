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

def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges
    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().
    Examples:
    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps

def span_compute_actual_possible(results: dict) -> dict:
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    correct = results['correct']
    incorrect = results['incorrect']
    partial = results['partial']
    missed = results['missed']
    spurious = results['spurious']

    # Possible: number annotations in the gold-standard which contribute to the
    # final score

    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system

    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results

def span_compute_precision_recall(results: dict, partial_or_type) -> dict:
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results['partial']
    correct = results['correct']

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall

    return results

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

    