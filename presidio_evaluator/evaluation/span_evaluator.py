from tqdm import tqdm
from copy import deepcopy
from typing import List, Optional, Dict
import numpy as np

from presidio_evaluator.models import BaseModel
from presidio_evaluator.evaluation import SpanEvaluationResult, SpanOutput
from presidio_evaluator import InputSample, Span


class SpanEvaluator:
    def __init__(
        self,
        model: BaseModel,
        entities_to_keep: Optional[List[str]] = None,
        overlap_threshold: float = 0.5
    ):
        """
        Evaluate a PII detection model at the span level

        :param model: Instance of a fitted model (of base type BaseModel)
        :param entities_to_keep: List of entity names to focus the evaluator on (and ignore the rest).
        Default is None = all entities. If the provided model has a list of entities to keep,
        this list would be used for evaluation.
        :param overlap_threshold: If the entity type of gold and predict spans are matched and 
        the overlapping ratio between the gold and the predicted span >= overlap_threshold, the predicted span is considered as a partial match
        Default is 0.5
        """
        self.model = model
        self.overlap_threshold = overlap_threshold
        self.entities_to_keep = entities_to_keep
        if self.entities_to_keep is None and self.model.entities:
            self.entities_to_keep = self.model.entities


    @staticmethod 
    def get_matched_gold(sample: InputSample, 
                        pred_span: Span, 
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
                             full_text=sample.full_text,
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
                                full_text=sample.full_text,
                                overlap_score=max_overlapping
                                )
            else:
                # Scenario 3: Entity types are wrong but the spans' boundaries overlapping ratio is between [overlap_threshold, 1]
                # output_type is partial
                return SpanOutput(output_type="partial",
                                gold_span=matched_gold_span,
                                pred_span=pred_span,
                                full_text=sample.full_text,
                                overlap_score=max_overlapping
                                )
        elif 0 < max_overlapping < overlap_threshold:
            # Senario 4: regardless of what the predicted entity is if the spans' boundaries overlapping ratio is between (0, overlap_threshold]
            # output_type is incorrect
            return SpanOutput(output_type="incorrect",
                             gold_span=matched_gold_span,
                             pred_span=pred_span,
                             full_text=sample.full_text,
                             overlap_score=max_overlapping
                             )
        else:
            # Scenario 5: Regardless of the entity type, spans' boundaries overlapping ratio is 0 
            # output_type is spurious
            return SpanOutput(output_type="spurious",
                             gold_span=None,
                             pred_span=pred_span,
                             full_text=sample.full_text,
                             overlap_score=max_overlapping
                             )
        

    @staticmethod
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

    @staticmethod
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


    def evaluate_span(self, dataset: List[InputSample]) -> SpanEvaluationResult:
        """
        Evaluate the dataset at span level
        """
        eval_output = {}
        eval_output["metrics"] = {}
        eval_output["span_output"] = {}
        evaluation = {"strict": 0, "exact": 0, "partial": 0, "incorrect": 0, "miss": 0, "spurious": 0}
        evaluate_by_entities_type = {e: deepcopy(evaluation) for e in self.entities_to_keep}
        # List of SpanEvaluatorOutput which holds the details of model erros for analysis purposes
        span_outputs = []

        for sample in tqdm(dataset, desc=f"Evaluating {self.model.__class__}"):
            # prediction
            response_spans = self.model.predict_span(sample)

            # span evaluator
            # filter gold and pred spans which their entities are in the list of entities_to_keep
            gold_spans = [ent for ent in sample.spans if ent.entity_type in self.entities_to_keep]
            pred_spans = [ent for ent in response_spans if ent.entity_type in self.entities_to_keep]

            # keep track of true entities that overlapped
            true_which_overlapped_with_pred = []

            for pred in pred_spans:
                model_output = self.get_matched_gold(sample, pred, gold_spans, self.overlap_threshold)
                output_type = model_output.output_type
                span_outputs.append(model_output)

                evaluation[output_type] += 1
                evaluate_by_entities_type[model_output.gold_span.entity_type] += 1
                
                if output_type in ['strict', 'exact', 'partial', 'incorrect']:
                    true_which_overlapped_with_pred.append(model_output.gold_span)
                
            ## Get all missed span/entity in the gold corpus
            for true in gold_spans:
                if true in true_which_overlapped_with_pred:
                    continue
                else:
                    evaluation["miss"] += 1
                    evaluate_by_entities_type[true.entity_type]["miss"] += 1
                    # Add the output's detail to evaluation_results
                    span_outputs.append(SpanOutput(
                            output_type = "miss",
                            gold_span = true,
                            overlap_score=0,
                            full_text=sample.full_text
                        ))
        # Compute overall "possible", "actual" and precision and recall 
        evaluation = self.compute_span_actual_possible(evaluation)
        # Compute actual and possible of each entity
        for entity_type in evaluate_by_entities_type:
            evaluate_by_entities_type[entity_type] = self.compute_span_actual_possible(evaluate_by_entities_type[entity_type])

        ### Generate final output of evaluation
        eval_output["span_output"]["global"] = evaluation
        eval_output["metrics"]['global'] = self.compute_precision_recall(evaluation)
        for type in evaluate_by_entities_type:
            eval_output["span_output"][type] = evaluate_by_entities_type[type]
            eval_output["metrics"][type] = self.compute_precision_recall(evaluate_by_entities_type[type])
        
        # Return output of SpanEvaluationResult format
        spans_eval = SpanEvaluationResult(span_outputs=span_outputs, model_metrics=eval_output)
        
        return spans_eval
