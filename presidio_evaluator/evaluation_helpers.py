import numpy as np
from typing import List

from presidio_evaluator import Span
from presidio_evaluator.evaluation import SpanOutput


def get_matched_gold(pred_span: Span, 
                        gold_span: List[Span], 
                        overlap_threshold) -> SpanOutput:
        """
        Given a pred_span, get the best matchest gold span based on the overlap_threshold. 
        Return a SpanOutput
        :param sample: InputSample
        :param pred_span: Span,  Predicted span
        :param gold_span: List[Span]: List of gold spans from the annotation input
        """
        max_overlapping = 0
        for gold in gold_span:
            if pred_span.__eq__(gold):
                return SpanOutput(output_type="strict",
                             gold_span=gold,
                             pred_span=pred_span,
                             overlap_score=1
                             )
            else:
                # Calculate the overlapping between gold and predicted span
                overlap_ratio = float(pred_span.intersect(gold, ignore_entity_type = True) / np.max([len(gold.entity_value), len(pred_span.entity_value)]))
                # Get highest overlapping ratio
                if overlap_ratio > max_overlapping:
                    max_overlapping = overlap_ratio
                    matched_gold_span = gold
        
        if max_overlapping >= overlap_threshold:
            if matched_gold_span.entity_type == pred_span.entity_type:
                # Scenario 2: Entity types match but the spans' boundaries overlapping ratio is between [overlap_threshold, 1)
                # output_type is exact
                return SpanOutput(output_type="exact",
                                gold_span=matched_gold_span,
                                pred_span=pred_span,
                                overlap_score=max_overlapping
                                )
            else:
                # Scenario 3: Entity types are wrong but the spans' boundaries overlapping ratio is between [overlap_threshold, 1]
                # output_type is partial
                return SpanOutput(output_type="partial",
                                gold_span=matched_gold_span,
                                pred_span=pred_span,
                                overlap_score=max_overlapping
                                )
        elif 0 < max_overlapping < overlap_threshold:
            # Senario 4: regardless of what the predicted entity is if the spans' boundaries overlapping ratio is between (0, overlap_threshold]
            # output_type is incorrect
            return SpanOutput(output_type="incorrect",
                             gold_span=matched_gold_span,
                             pred_span=pred_span,
                             overlap_score=max_overlapping
                             )
        else:
            # Scenario 5: Regardless of the entity type, spans' boundaries overlapping ratio is 0 
            # output_type is spurious
            return SpanOutput(output_type="spurious",
                             gold_span=None,
                             pred_span=pred_span,
                             overlap_score=max_overlapping
                             )

def compute_span_actual_possible(results: dict) -> dict:
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

def compute_precision_recall(results: dict) -> dict:
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