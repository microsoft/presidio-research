from tqdm import tqdm
from copy import deepcopy
from typing import List, Optional, Dict
from difflib import SequenceMatcher

from presidio_evaluator.models import BaseModel
from presidio_evaluator.evaluation import SpanEvaluationResult, SpanError
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
    def is_overlap(gold_span: Span, pred_span: Span, overlap_threshold = 0.5):
        """
        Calculate the overlap between the gold boundary and predicted boundary. Return True if the actual overlap ratio >= 0.5

        :param gold_span: Gold span from the annotation input
        :param pred_span: Predicted span
        """
        overlap_ratio = SequenceMatcher(None, gold_span.entity_value, pred_span.entity_value).ratio()
        if overlap_ratio >= overlap_threshold:
            return True
        else:
            return False

    @staticmethod
    def compute_span_actual_possible(results):
        """
        Take the result dict and calculate the actual and possible spans
        """
        correct = results["correct"]
        incorrect = results["incorrect"]
        partial = results["partial"]
        missed = results["miss"]
        spurious = results["spurious"]
        # Possible: Number of annotations in the gold-standard which contribute to the final score
        possible = correct + incorrect + partial + missed
        # Actual: Number of annotations produced by the PII detection system
        actual = correct + incorrect + partial + spurious

        results["actual"] = actual
        results["possible"] = possible
        
        return results

    @staticmethod
    def compute_precision_recall(results):
        """
        Take the result dict to calculate the strict and flexible precision/ recall
        """
        metrics = {}
        correct = results["correct"]
        partial = results["partial"]
        actual = results["actual"]
        possible = results["possible"]
        # Calculate the partial performance
        precision_flexible = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall_flexible = (correct + 0.5 * partial) / possible if possible > 0 else 0
        # Calculate the partial performance
        precision_strict = correct / actual if actual > 0 else 0
        recall_strict = correct / possible if possible > 0 else 0

        metrics["precision_strict"] = precision_strict
        metrics["recall_strict"] = recall_strict
        metrics["precision_flexible"] = precision_flexible
        metrics["recall_flexible"] = recall_flexible
        return metrics

    
    def evaluate_span(self, dataset: List[InputSample]) -> SpanEvaluationResult:
        """
        Evaluate the dataset at span level
        """
        eval_output = {}
        eval_output["metrics"] = {}
        eval_output["span_output"] = {}
        evaluation = {"correct": 0, "partial": 0, "incorrect": 0, "miss": 0, "spurious": 0}
        evaluate_by_entities_type = {e: deepcopy(evaluation) for e in self.entities_to_keep}
        # List of SpanEvaluatorOutput which holds the details of model erros for analysis purposes
        model_errors = []
        
        
        for sample in tqdm(dataset, desc=f"Evaluating {self.model.__class__}"):
            # prediction
            response_spans = self.model.predict_span(sample)

            # span evaluator
            # filter gold and pred spans which their entities are in the list of entities_to_keep
            gold_named_entities = [ent for ent in sample.spans if ent.entity_type in self.entities_to_keep]
            pred_named_entities = [ent for ent in response_spans if ent.entity_type in self.entities_to_keep]

            # keep track of entities that overlapped
            true_which_overlapped_with_pred = []

            for pred in pred_named_entities:
                if pred in gold_named_entities:
                    # strict match
                    evaluation["correct"] += 1
                    evaluate_by_entities_type[pred.entity_type]["correct"] += 1
                    true_which_overlapped_with_pred.append(pred)
                else:
                    # Check if there is a partial match between gold and pred span
                    for gold in gold_named_entities:
                        # Overlap between span, entity is match
                        if pred.entity_type == gold.entity_type: 
                            overlap_ratio = self.is_overlap(gold, pred, self.overlap_threshold)
                            if overlap_ratio:
                                evaluation["partial"] += 1
                                evaluate_by_entities_type[pred.entity_type]["partial"] += 1
                                true_which_overlapped_with_pred.append(gold)
                                # Add the output's detail to evaluation_results
                                model_errors.append(SpanError(
                                        error_type = "partial",
                                        gold_span = gold,
                                        pred_span = pred,
                                        overlap_score=overlap_ratio,
                                        full_text=sample.full_text
                                    ))
                            else:
                                # Entity type is correct but the overlap ratio is smaller than 0.5
                                evaluation["incorrect"] += 1
                                evaluate_by_entities_type[pred.entity_type]["incorrect"] += 1
                                true_which_overlapped_with_pred.append(gold)
                                # Add the output's detail to evaluation_results
                                model_errors.append(SpanError(
                                        error_type = "incorrect",
                                        gold_span = gold,
                                        pred_span = pred,
                                        overlap_score=overlap_ratio,
                                        full_text=sample.full_text
                                    ))
                        else: 
                            # Entity type is incorrect 
                            evaluation["spurious"] += 1
                            evaluate_by_entities_type[pred.entity_type]["spurious"] += 1
                            # Add the output's detail to evaluation_results
                            model_errors.append(SpanError(
                                    error_type = "spurious",
                                    gold_span = gold,
                                    pred_span = pred,
                                    overlap_score=0,
                                    full_text=sample.full_text
                                ))

            ## Get all missed span/entity in the gold corpus
            for true in gold_named_entities:
                if true in true_which_overlapped_with_pred:
                    continue
                else:
                    evaluation["miss"] += 1
                    evaluate_by_entities_type[true.entity_type]["miss"] += 1
                    # Add the output's detail to evaluation_results
                    model_errors.append(SpanError(
                            error_type = "miss",
                            gold_span = gold,
                            pred_span = pred,
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
        eval_output["metrics"]['global_span_metrics'] = self.compute_precision_recall(evaluation)
        for type in evaluate_by_entities_type:
            eval_output["span_output"][type] = evaluate_by_entities_type[type]
            eval_output["metrics"][f'{type}_span_metrics'] = self.compute_precision_recall(evaluate_by_entities_type[type])
        
        # Return output of SpanEvaluationResult format
        spans_eval = SpanEvaluationResult(model_errors=model_errors, model_metrics=eval_output)
        
        return spans_eval
