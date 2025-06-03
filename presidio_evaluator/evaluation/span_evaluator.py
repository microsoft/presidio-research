from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict
from presidio_evaluator.evaluation.skipwords import get_skip_words
from presidio_evaluator.data_objects import Span


@dataclass
class EntityTypeMetrics:
    """Metrics for a specific entity type."""
    precision: float
    recall: float
    f_beta: float
    num_predicted: int
    num_annotated: int
    true_positives: int
    false_positives: int
    false_negatives: int

@dataclass
class SpanEvaluationResult:
    """Results of span-based evaluation."""
    precision: float
    recall: float
    f_beta: float
    total_predicted: int
    total_annotated: int
    total_true_positives: int
    total_false_positives: int
    total_false_negatives: int
    per_type: Dict[str, EntityTypeMetrics]
    error_analysis: Dict[str, int]
    entity_type_mismatches: List[Dict[str, str]]

    def to_dict(self) -> Dict:
        """Convert evaluation results to dictionary format for backward compatibility."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f_beta": self.f_beta,
            "per_type": {
                entity_type: {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f_beta": metrics.f_beta,
                }
                for entity_type, metrics in self.per_type.items()
            }
        }

class SpanEvaluator:
    """
    Evaluates PII detection using span-based fuzzy matching with token-level Intersection over Union (IoU).
    """

    def __init__(
        self,
        iou_threshold: float = 0.9,
        schema: str = None,
        beta: int = 2,
        skip_words: Optional[List] = None,
        merge_adjacent_spans: bool = True
    ):
        """
        Initialize the SpanEvaluator for evaluating pii entities detection results.

        :param iou_threshold: Minimum Intersection over Union (IoU) threshold for considering spans as matching.
                            Value between 0 and 1, where higher values require more overlap (default: 0.5)
        :param skip_words: Optional list of custom skip words to ignore during token normalization,
                            should also include punctuation marks.
                         If None, uses skip words from skipwords.py (default: None).
                         Pass an empty list ([]) to disable skip word removal entirely.
        :param schema: The labeling schema to use for span creation. Valid values:
                      - 'BIO': Use Begin/Inside/Outside labeling scheme
                      - None: Use default scheme with entity start indicators (default: None)
        :param beta: The beta parameter for F-beta score calculation. Default is 2.
        :param merge_adjacent_spans: Whether to merge adjacent spans of the same entity type. 
                                    Spans are considered adjacent if they are separated only by skip words or punctuation.
                                    Default is True.
        """
        self.iou_threshold = iou_threshold
        self.schema = schema
        self.beta = beta
        self.merge_adjacent_spans = merge_adjacent_spans
        self.skip_words = skip_words if skip_words else get_skip_words()
        
    def _normalize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Normalize tokens by:
        1. Converting to lowercase
        2. Removing stop words
        3. Removing standalone punctuation
        4. Removing skip words (common words that shouldn't affect entity matching)

        :param tokens: List of token strings to normalize
        :return: List of normalized tokens
        """
        normalized = []
        for token in tokens:
            token = token.lower()
            # Skip if token is in skip words list
            if token in self.skip_words:
                continue
            normalized.append(token)
        return normalized

    def _merge_adjacent_spans(self, spans: List[Span], df: pd.DataFrame) -> List[Span]:
        """
        Merge adjacent spans of the same entity type if separated only by skip words / punctuation.

        :param spans: List of Span objects to potentially merge
        :param df: DataFrame containing the tokens and their positions
        :return: List of merged Span objects
        """
        if not spans:
            return []
        spans = sorted(spans, key=lambda x: x.start_position)
        merged = []
        current = spans[0]

        for next_span in spans[1:]:
            if (
                current.entity_type == next_span.entity_type
                and self._are_spans_adjacent(current, next_span, df)
            ):
                merged_tokens = current.entity_value + next_span.entity_value
                current = Span(
                    entity_type=current.entity_type,
                    entity_value=merged_tokens,
                    start_position=current.start_position,
                    end_position=next_span.end_position,
                    normalized_value=self._normalize_tokens(merged_tokens)
                )
            else:
                merged.append(current)
                current = next_span
        
        merged.append(current)
        return merged

    def _are_spans_adjacent(self, span1: Span, span2: Span, df: pd.DataFrame) -> bool:
        """
        Check if two spans are adjacent, i.e., separated only by skipwords / punctuation or whitespace tokens.

        :param span1: First Span object
        :param span2: Second Span object
        :param df: DataFrame containing the tokens
        :return: True if spans are adjacent, False otherwise
        """
        # Slice tokens between span1 and span2 using the row indices
        between_tokens = df.loc[
            span1.end_position : span2.start_position - 1, "token"
        ].tolist()
        non_skip_tokens = [tok for tok in between_tokens if tok not in self.skip_words]
        return len(non_skip_tokens) == 0

    @staticmethod
    def calculate_iou(span1: Span, span2: Span) -> float:
        """
        Calculate the Intersection over Union (IoU) between two spans using normalized tokens.

        :param span1: First Span object
        :param span2: Second Span object
        :param df: DataFrame containing the tokens
        :return: IoU score (float between 0 and 1)
        """
        set1 = set("".join(span1.normalized_value))
        set2 = set("".join(span2.normalized_value))

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0
    
    def _initialize_metrics(self):
        return {
            "total_true_positives": 0,
            "total_false_positives": 0,
            "total_false_negatives": 0,
            "total_num_annotated": 0,
            "total_num_predicted": 0,
            "per_type_metrics": defaultdict(
                lambda: {"tp": 0, "fp": 0, "fn": 0, "num_annotated": 0, "num_predicted": 0}
            ),
            "error_analysis": defaultdict(int)
        }
    
    def _process_sentence_spans(self, sentence_df):
        annotation_spans = self._create_spans(sentence_df, "annotation")
        prediction_spans = self._create_spans(sentence_df, "prediction")
        
        if self.merge_adjacent_spans:
            annotation_spans = self._merge_adjacent_spans(annotation_spans, sentence_df)
            prediction_spans = self._merge_adjacent_spans(prediction_spans, sentence_df)
            
        return annotation_spans, prediction_spans
    
    def _handle_unmatched_predictions(self, prediction_spans, matched_preds, metrics):
        """
        Handle predictions that weren't matched to any annotation.
        
        Args:
            prediction_spans: List of prediction Span objects
            matched_preds: Set of already matched prediction spans
            metrics: Dictionary containing the metrics to update
        """
        for pred_span in prediction_spans:
            pred_span_key = (
                pred_span.entity_type,
                pred_span.start_position,
                pred_span.end_position,
            )
            if pred_span_key not in matched_preds:
                metrics["total_false_positives"] += 1
                metrics["per_type_metrics"][pred_span.entity_type]["fp"] += 1
                metrics["error_analysis"][f"extra_{pred_span.entity_type}"] += 1

    def _is_valid_prediction(self, pred_span: Span, ann_span: Span, matched_preds: set) -> bool:
        """
        Check if a prediction span is valid for matching with an annotation span.
        
        A prediction is valid if:
        1. It hasn't already been matched to another annotation
        2. Its entity type matches the annotation's entity type
        
        Args:
            pred_span: The prediction Span to check
            ann_span: The annotation Span being matched against
            matched_preds: Set of already matched prediction spans
        
        Returns:
            bool: True if the prediction is valid for matching, False otherwise
        """
        # Create unique key for the prediction span
        pred_span_key = (
            pred_span.entity_type,
            pred_span.start_position,
            pred_span.end_position,
        )
        
        # Check if prediction is already matched
        if pred_span_key in matched_preds:
            return False
            
        # Check if the prediction is a non-entity (O) while the annotation is an entity
        if pred_span.entity_type == "O" and ann_span.entity_type != "O":
            return False
            
        return True

    def _find_best_match(self, ann_span, prediction_spans, matched_preds):
        best_match = None
        best_iou = 0.0
        
        for pred_span in prediction_spans:
            if self._is_valid_prediction(pred_span, ann_span, matched_preds):
                iou = self.calculate_iou(ann_span, pred_span)
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_span
                    
        return best_match, best_iou


    def _create_evaluation_result(self, metrics):
        global_metrics = self._calculate_metrics(
            metrics["total_true_positives"],
            metrics["total_num_predicted"],
            metrics["total_num_annotated"],
            self.beta
        )
        
        per_type_results = self._calculate_per_type_metrics(metrics["per_type_metrics"])
        
        return SpanEvaluationResult(
            precision=global_metrics["precision"],
            recall=global_metrics["recall"],
            f_beta=global_metrics["f_beta"],
            total_predicted=metrics["total_num_predicted"],
            total_annotated=metrics["total_num_annotated"],
            total_true_positives=metrics["total_true_positives"],
            total_false_positives=metrics["total_false_positives"],
            total_false_negatives=metrics["total_false_negatives"],
            per_type=per_type_results,
            error_analysis=dict(metrics["error_analysis"]),
            entity_type_mismatches=metrics.get("entity_type_mismatches", [])
        )


    def _match_predictions_with_annotations(self, annotation_spans, prediction_spans, metrics):
        """
        Match predictions to annotations and update metrics accordingly.
        
        A true positive is counted when:
        - A prediction matches an annotation with IoU >= threshold
        - The prediction hasn't been matched to another annotation yet
        
        Args:
            annotation_spans: List of annotation Span objects
            prediction_spans: List of prediction Span objects
            metrics: Dictionary containing metrics to update
        """
        matched_preds = set()
        matched_anns = set()
        entity_type_mismatches = []
        
        # Sort predictions by IoU score to ensure we match the best predictions first
        all_matches = []
        for ann_span in annotation_spans:
            for pred_span in prediction_spans:
                iou = self.calculate_iou(ann_span, pred_span)
                if iou >= self.iou_threshold:
                    all_matches.append((ann_span, pred_span, iou))
        
        # Sort matches by IoU score in descending order
        all_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Process matches in order of highest IoU
        for ann_span, pred_span, iou in all_matches:
            pred_key = (pred_span.entity_type, pred_span.start_position, pred_span.end_position)
            ann_key = (ann_span.entity_type, ann_span.start_position, ann_span.end_position)
            
            # Skip if either span is already matched
            if pred_key in matched_preds or ann_key in matched_anns:
                continue
            
            # Count as true positive
            metrics["total_true_positives"] += 1
            
            # Handle type matching/mismatching
            if pred_span.entity_type == ann_span.entity_type:
                metrics["per_type_metrics"][ann_span.entity_type]["tp"] += 1
            else:
                # Record type mismatch
                mismatch = {
                    "true_type": ann_span.entity_type,
                    "predicted_type": pred_span.entity_type,
                    "text": " ".join(pred_span.entity_value),
                    "start": pred_span.start_position,
                    "end": pred_span.end_position,
                    "iou": iou
                }
                entity_type_mismatches.append(mismatch)
                metrics["error_analysis"][f"type_mismatch_{ann_span.entity_type}_as_{pred_span.entity_type}"] += 1
                
                # Update per-type metrics
                metrics["per_type_metrics"][ann_span.entity_type]["fn"] += 1
                metrics["per_type_metrics"][pred_span.entity_type]["fp"] += 1
            
            matched_preds.add(pred_key)
            matched_anns.add(ann_key)
        
        # Handle unmatched annotations as false negatives
        for ann_span in annotation_spans:
            ann_key = (ann_span.entity_type, ann_span.start_position, ann_span.end_position)
            if ann_key not in matched_anns:
                metrics["total_false_negatives"] += 1
                metrics["per_type_metrics"][ann_span.entity_type]["fn"] += 1
                metrics["error_analysis"][f"missed_{ann_span.entity_type}"] += 1
        
        # Handle unmatched predictions as false positives
        for pred_span in prediction_spans:
            pred_key = (pred_span.entity_type, pred_span.start_position, pred_span.end_position)
            if pred_key not in matched_preds:
                metrics["total_false_positives"] += 1
                metrics["per_type_metrics"][pred_span.entity_type]["fp"] += 1
                metrics["error_analysis"][f"extra_{pred_span.entity_type}"] += 1
        
        metrics["entity_type_mismatches"] = entity_type_mismatches


    def _calculate_per_type_metrics(self, per_type_metrics: dict) -> Dict[str, EntityTypeMetrics]:
        """
        Calculate precision, recall, and F-beta scores for each entity type.
        
        Args:
            per_type_metrics: Dictionary containing per-type counts of true positives,
                            false positives, false negatives, and total counts
        
        Returns:
            Dictionary mapping entity types to EntityTypeMetrics objects
        """
        per_type_results = {}
        
        for entity_type, counts in per_type_metrics.items():
            # Calculate metrics for this entity type
            metrics = self._calculate_metrics(
                counts["tp"],
                counts["num_predicted"],
                counts["num_annotated"],
                self.beta
            )
            
            # Create EntityTypeMetrics object
            per_type_results[entity_type] = EntityTypeMetrics(
                precision=metrics["precision"],
                recall=metrics["recall"],
                f_beta=metrics["f_beta"],
                num_predicted=counts["num_predicted"],
                num_annotated=counts["num_annotated"],
                true_positives=counts["tp"],
                false_positives=counts["fp"],
                false_negatives=counts["fn"]
            )
        
        return per_type_results


    def _update_per_type_counts(self, annotation_spans, prediction_spans, per_type_metrics):
        """
        Update the per-entity type counts for annotations and predictions.
        
        Args:
            annotation_spans: List of annotation Span objects
            prediction_spans: List of prediction Span objects
            per_type_metrics: Dictionary to track per-type metrics
        """
        for ann_span in annotation_spans:
            per_type_metrics[ann_span.entity_type]["num_annotated"] += 1
        for pred_span in prediction_spans:
            per_type_metrics[pred_span.entity_type]["num_predicted"] += 1

    
    def evaluate(self, results_df: pd.DataFrame) -> SpanEvaluationResult:
        """
        Evaluate the predictions against ground truth annotations.
        
        This method orchestrates the evaluation process by:
        1. Initializing evaluation metrics
        2. Processing each sentence to get annotation and prediction spans
        3. Matching predictions with annotations and tracking metrics
        4. Creating and returning the final evaluation result
        
        Args:
            results_df: DataFrame containing tokens, annotations and predictions
            
        Returns:
            SpanEvaluationResult object containing precision, recall, f_beta and per-type metrics
        """
        # Initialize metrics tracking structures
        metrics = self._initialize_metrics()
        
        # Process each sentence
        for _, sentence_df in results_df.groupby("sentence_id"):
            # Get and process spans for the sentence
            annotation_spans, prediction_spans = self._process_sentence_spans(sentence_df)
            
            # Update total counts
            metrics["total_num_annotated"] += len(annotation_spans)
            metrics["total_num_predicted"] += len(prediction_spans)
            
            # Update per-entity type counts
            self._update_per_type_counts(
                annotation_spans, 
                prediction_spans, 
                metrics["per_type_metrics"]
            )
            
            # Match predictions with annotations and update metrics
            self._match_predictions_with_annotations(
                annotation_spans,
                prediction_spans,
                metrics
            )
        
        # Create and return the final evaluation result
        return self._create_evaluation_result(metrics)
        

    def _create_spans(self, df: pd.DataFrame, column: str) -> List[Span]:
        """
        Create spans from a DataFrame column.
        
        Args:
            df (pd.DataFrame): DataFrame containing the spans.
            column (str): Name of the column to extract spans from.
            
        Returns:
            List[Span]: List of Span objects created from the DataFrame.
        """
        if self.schema == "BIO":
            return self._create_spans_bio(df, column)

        spans = []
        current_entity_type = None
        current_tokens = []
        current_start = None

        for idx, row in df.iterrows():
            entity_type = row[column]
            token = row["token"]
            is_entity_start = row["start_indices"]
            # if token in self.skip_words:
            #     entity_type = "O"  # Treat skip words as non-entities
            
            if entity_type == "O":
                if current_entity_type and current_tokens:
                    normalized_tokens = self._normalize_tokens(current_tokens)
                    if normalized_tokens:
                        spans.append(
                            Span(
                                entity_type=current_entity_type,
                                entity_value=current_tokens,
                                start_position=current_start,
                                end_position=idx,
                                normalized_value=normalized_tokens
                            )
                        )
                    current_entity_type = None
                    current_tokens = []
                    current_start = None
                continue

            if (entity_type != current_entity_type) or (is_entity_start and column == "annotation"):
                if current_entity_type and current_tokens:
                    normalized_tokens = self._normalize_tokens(current_tokens)
                    if normalized_tokens:
                        spans.append(
                            Span(
                                entity_type=current_entity_type,
                                entity_value=current_tokens,
                                start_position=current_start,
                                end_position=idx,
                                normalized_value=normalized_tokens
                            )
                        )
                current_entity_type = entity_type
                current_tokens = [token]
                current_start = idx
        
            else:
                current_tokens.append(row["token"])

        # Handle final span
        if current_entity_type and current_tokens:
            normalized_tokens = self._normalize_tokens(current_tokens)
            if normalized_tokens:
                spans.append(
                    Span(
                        entity_type=current_entity_type,
                        entity_value=current_tokens,
                        start_position=current_start,
                        end_position=df.index[-1] + 1,
                        normalized_value=normalized_tokens
                    )
                )
        return spans
       

    @staticmethod
    def _calculate_metrics(
        true_positives: int, num_predicted: int, num_annotated: int, beta: int = 2
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F-beta score using the new logic.

        :param true_positives: Number of true positives
        :param num_predicted: Number of predicted spans
        :param num_annotated: Number of annotated (gold) spans
        :param beta: The beta parameter for F-beta score calculation. Default is 2.
        :return: Dictionary containing precision, recall, and f-beta metrics
        """
        precision = (
            true_positives / num_predicted if num_predicted > 0 else 0.0
        )
        recall = (
            true_positives / num_annotated if num_annotated > 0 else 0.0
        )
        f_beta = (
            (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f_beta": f_beta}

    def span_pairwise_iou_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each sentence, compute IoU for all combinations of annotated and predicted spans.
        Returns a DataFrame with columns:
        ['sentence_id', 'ann_entity', 'ann_start', 'ann_end', 'pred_entity', 'pred_start', 'pred_end', 'iou']

        :param df: DataFrame containing sentence tokens and entity annotations/predictions
        :return: DataFrame with IoU scores for all span pairs per sentence
        """
        records = []
        for sentence_id, sentence_df in df.groupby("sentence_id"):
            annotation_spans, prediction_spans = self._process_sentence_spans(sentence_df)
            for ann in annotation_spans:
                for pred in prediction_spans:
                    iou = self.calculate_iou(ann, pred)
                    records.append(
                        {
                            "sentence_id": sentence_id,
                            "ann_entity": ann.entity_type,
                            "ann_start": ann.start_position,
                            "ann_end": ann.end_position,
                            "pred_entity": pred.entity_type,
                            "pred_start": pred.start_position,
                            "pred_end": pred.end_position,
                            "iou": iou,
                        }
                    )
        return pd.DataFrame(records)

    def _create_spans_bio(self, df: pd.DataFrame, column: str) -> List[Span]:
        """
        Create Span objects from a DataFrame using BIO labeling scheme with normalization.

        :param df: DataFrame containing tokens and BIO labels
        :param column: Column name containing entity labels ('annotation' or 'prediction')
        :return: List of Span objects
        """
        spans = []
        current_type = None
        current_tokens = []
        current_start = None

        for idx, row in df.iterrows():
            tag = row[column]

            # Handle non-entity tokens
            if tag == "O":
                if current_type:
                    normalized_tokens = self._normalize_tokens(current_tokens)
                    if normalized_tokens:  # Only create span if we have tokens after normalization
                        spans.append(
                            Span(
                                entity_type=current_type,
                                entity_value=current_tokens,  # Keep original tokens
                                start_position=current_start,
                                end_position=idx,
                                normalized_value=normalized_tokens  # Add normalized tokens
                            )
                        )
                    current_type = None
                    current_tokens = []
                    current_start = None
                continue

            # Split BIO tag into B/I and entity type
            bio_prefix, entity_type = tag.split("-", 1)

            # Start of new entity
            if bio_prefix == "B":
                if current_type:
                    normalized_tokens = self._normalize_tokens(current_tokens)
                    if normalized_tokens:
                        spans.append(
                            Span(
                                entity_type=current_type,
                                entity_value=current_tokens,
                                start_position=current_start,
                                end_position=idx,
                                normalized_value=normalized_tokens
                            )
                        )
                current_type = entity_type
                current_tokens = [row["token"]]
                current_start = idx

            # Inside of entity
            elif bio_prefix == "I" and current_type == entity_type:
                current_tokens.append(row["token"])

        # Add final span if exists
        if current_type:
            normalized_tokens = self._normalize_tokens(current_tokens)
            if normalized_tokens:
                spans.append(
                    Span(
                        entity_type=current_type,
                        entity_value=current_tokens,
                        start_position=current_start,
                        end_position=df.index[-1] + 1,
                        normalized_value=normalized_tokens
                    )
                )

        return spans

    @staticmethod
    def convert_to_bio_scheme(df: pd.DataFrame, entity_column: str) -> pd.Series:
        """
        Convert entity annotations to BIO scheme.

        Args:
            df: DataFrame with token-level annotations
            entity_column: Name of column containing entity labels

        Returns:
            Series with BIO tags
        """
        bio_tags = []
        prev_entity = "O"

        for idx, row in df.iterrows():
            current_entity = row[entity_column]

            if current_entity == "O":
                bio_tags.append("O")
            elif prev_entity != current_entity:
                # Start of new entity
                bio_tags.append(f"B-{current_entity}")
            else:
                # Inside existing entity
                bio_tags.append(f"I-{current_entity}")

            prev_entity = current_entity

        return pd.Series(bio_tags)
