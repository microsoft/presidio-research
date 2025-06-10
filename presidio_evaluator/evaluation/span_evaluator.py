from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict

from presidio_analyzer import AnalyzerEngine

from presidio_evaluator.evaluation import BaseEvaluator, EvaluationResult
from presidio_evaluator.data_objects import Span
from evaluation_result import EvaluationResult
from presidio_evaluator.models import BaseModel


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
class SpanEvaluationResult(EvaluationResult):
    """Results of span-based evaluation."""

    # precision: float
    # recall: float
    # f_beta: float
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
            },
        }



class SpanEvaluator(BaseEvaluator):
    """
    Evaluates PII detection using span-based fuzzy matching with character-level Intersection over Union (IoU).
    """

    def __init__(
        self,
        model: Union[BaseModel, AnalyzerEngine],
        verbose: bool = False,
        compare_by_io=True,
        entities_to_keep: Optional[List[str]] = None,
        generic_entities: Optional[List[str]] = None,
        skip_words: Optional[List] = None,
        iou_threshold: float = 0.9,
        merge_adjacent_spans: bool = True,
        char_based: bool = True,
    ):
        """
        Initialize the SpanEvaluator for evaluating pii entities detection results.

        :param iou_threshold: Minimum Intersection over Union (IoU) threshold for considering spans as matching.
                            Value between 0 and 1, where higher values require more overlap (default: 0.5)
        :param skip_words: Optional list of custom skip words to ignore during token normalization,
                            should also include punctuation marks.
                         If None, uses skip words from skipwords.py (default: None).
                         Pass an empty list ([]) to disable skip word removal entirely.
        :param merge_adjacent_spans: Whether to merge adjacent spans of the same entity type.
                                    Spans are considered adjacent if they are separated only by skip words or punctuation.
                                    Default is True.
        :param char_based: If True, calculate IoU at the character-level, else, calculate iou at the token-level.
        """
        super().__init__(
            model=model,
            verbose=verbose,
            compare_by_io=compare_by_io,
            entities_to_keep=entities_to_keep,
            generic_entities=generic_entities,
            skip_words=skip_words,
        )

        self.iou_threshold = iou_threshold
        self.merge_adjacent_spans = merge_adjacent_spans
        self.char_based = char_based

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
                    normalized_value=self._normalize_tokens(merged_tokens),
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
            span1.token_end : span2.token_start - 1, "token"
        ].tolist()
        non_skip_tokens = [tok for tok in between_tokens if tok not in self.skip_words]
        return len(non_skip_tokens) == 0

    @staticmethod
    def calculate_iou(span1: Span, span2: Span, char_based: bool) -> float:
        """
        Calculate the Intersection over Union (IoU) between two spans at character or token level.

        Args:
            span1: First Span object
            span2: Second Span object
            char_based: If True, calculate IoU based on character positions, else token positions

        Returns:
            IoU score (float between 0 and 1)
        """
        if char_based:
            # Get character ranges for both spans
            range1 = set(range(span1.start_position, span1.end_position))
            range2 = set(range(span2.start_position, span2.end_position))

            # Calculate intersection and union of character positions
            intersection = len(range1.intersection(range2))
            union = len(range1.union(range2))

            return intersection / union if union > 0 else 0.0
        else:
            # Token-based IoU calculation using token positions
            if (
                span1.token_start is None
                or span1.token_end is None
                or span2.token_start is None
                or span2.token_end is None
            ):
                # Fallback to normalized token comparison if token positions not available
                set1 = set(span1.normalized_value)
                set2 = set(span2.normalized_value)
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))

                return intersection / union if union > 0 else 0.0

            # Use token position ranges for IoU calculation
            range1 = set(range(span1.token_start, span1.token_end + 1))
            range2 = set(range(span2.token_start, span2.token_end + 1))

            # Calculate intersection and union of token positions
            intersection = len(range1.intersection(range2))
            union = len(range1.union(range2))

            return intersection / union if union > 0 else 0.0

    @staticmethod
    def _initialize_metrics():
        return {
            "total_true_positives": 0,
            "total_false_positives": 0,
            "total_false_negatives": 0,
            "total_num_annotated": 0,
            "total_num_predicted": 0,
            "per_type_metrics": defaultdict(
                lambda: {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "num_annotated": 0,
                    "num_predicted": 0,
                }
            ),
            "error_analysis": defaultdict(int),
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

        :param prediction_spans: List of prediction Span objects
        :param matched_preds: Set of already matched prediction spans
        :param metrics: Dictionary containing the metrics to check

        :returns: Dictionary with updates for false positives and error analysis
        """
        updates = {
            "total_false_positives": 0,
            "per_type_fp": defaultdict(int),
            "error_analysis": defaultdict(int),
        }

        for pred_span in prediction_spans:
            pred_span_key = (
                pred_span.entity_type,
                pred_span.start_position,
                pred_span.end_position,
            )
            if pred_span_key not in matched_preds:
                updates["total_false_positives"] += 1
                updates["per_type_fp"][pred_span.entity_type] += 1
                updates["error_analysis"][f"extra_{pred_span.entity_type}"] += 1

        return updates

    def _check_if_matched_already(
        self, pred_span: Span, ann_span: Span, matched_preds: set
    ) -> bool:
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

        # # Check if the prediction is a non-entity (O) while the annotation is an entity
        # if pred_span.entity_type == "O" and ann_span.entity_type != "O":
        #     return False

        return True

    def _find_best_match(self, ann_span, prediction_spans, matched_preds):
        best_match = None
        best_iou = 0.0

        for pred_span in prediction_spans:
            if self._check_if_matched_already(pred_span, ann_span, matched_preds):
                iou = self.calculate_iou(ann_span, pred_span, self.char_based)
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_span

        return best_match, best_iou

    def _create_evaluation_result(self, metrics, beta: int):
        global_metrics = self._calculate_metrics(
            metrics["total_true_positives"],
            metrics["total_num_predicted"],
            metrics["total_num_annotated"],
            beta,
        )

        per_type_results = self._calculate_per_type_metrics(metrics["per_type_metrics"], beta)

        return SpanEvaluationResult(
            pii_precision=global_metrics["precision"],
            recall=global_metrics["recall"],
            f_beta=global_metrics["f_beta"],
            total_predicted=metrics["total_num_predicted"],
            total_annotated=metrics["total_num_annotated"],
            total_true_positives=metrics["total_true_positives"],
            total_false_positives=metrics["total_false_positives"],
            total_false_negatives=metrics["total_false_negatives"],
            per_type=per_type_results,
            error_analysis=dict(metrics["error_analysis"]),
            entity_type_mismatches=metrics.get("entity_type_mismatches", []),
        )

    def _match_predictions_with_annotations(
        self, annotation_spans, prediction_spans, metrics
    ):
        """
        Match predictions to annotations and calculate metrics.

        Args:
            annotation_spans: List of annotation Span objects
            prediction_spans: List of prediction Span objects
            metrics: Dictionary containing current metrics state

        Returns:
            Dictionary with matching results and metric updates
        """
        matched_preds = set()
        entity_type_mismatches = []
        updates = {
            "total_true_positives": 0,
            "total_false_negatives": 0,
            "per_type_tp": defaultdict(int),
            "per_type_fn": defaultdict(int),
            "per_type_fp": defaultdict(int),
            "error_analysis": defaultdict(int),
        }

        # Process each annotation and find its best matching prediction
        for ann_span in annotation_spans:
            # Todo: check if it is needed to keep the matched predictions to avoid additional matches
            best_match, best_iou = self._find_best_match(
                ann_span, prediction_spans, matched_preds
            )

            if best_match and best_iou >= self.iou_threshold:
                # Count as true positive
                updates["total_true_positives"] += 1

                # Handle type matching/mismatching
                if best_match.entity_type == ann_span.entity_type:
                    updates["per_type_tp"][ann_span.entity_type] += 1
                else:
                    # Record type mismatch
                    mismatch = {
                        "true_type": ann_span.entity_type,
                        "predicted_type": best_match.entity_type,
                        "text": " ".join(best_match.entity_value),
                        "start": best_match.start_position,
                        "end": best_match.end_position,
                        "iou": best_iou,
                    }
                    entity_type_mismatches.append(mismatch)
                    updates["error_analysis"][
                        f"type_mismatch_{ann_span.entity_type}_as_{best_match.entity_type}"
                    ] += 1

                    # Update per-type metrics
                    updates["per_type_fn"][ann_span.entity_type] += 1
                    updates["per_type_fp"][best_match.entity_type] += 1

                # Mark prediction as matched
                matched_preds.add(
                    (
                        best_match.entity_type,
                        best_match.start_position,
                        best_match.end_position,
                    )
                )
            else:
                # No match found - false negative
                updates["total_false_negatives"] += 1
                updates["per_type_fn"][ann_span.entity_type] += 1
                if best_match:
                    updates["error_analysis"][f"low_iou_{ann_span.entity_type}"] += 1
                else:
                    updates["error_analysis"][f"missed_{ann_span.entity_type}"] += 1

        # Handle unmatched predictions as false positives
        unmatched_updates = self._handle_unmatched_predictions(
            prediction_spans, matched_preds, metrics
        )

        # Merge unmatched prediction updates
        updates["total_false_positives"] = unmatched_updates["total_false_positives"]
        for entity_type, count in unmatched_updates["per_type_fp"].items():
            updates["per_type_fp"][entity_type] += count
        for error_type, count in unmatched_updates["error_analysis"].items():
            updates["error_analysis"][error_type] += count

    def _calculate_per_type_metrics(
        self, per_type_metrics: dict, beta: int
    ) -> Dict[str, EntityTypeMetrics]:
        """
        Calculate precision, recall, and F-beta scores for each entity type.

        :param per_type_metrics: Dictionary containing per-type counts of true positives,
                            false positives, false negatives, and total counts
        :param beta: F-beta parameter

        :returns: Dictionary mapping entity types to EntityTypeMetrics objects
        """
        per_type_results = {}

        for entity_type, counts in per_type_metrics.items():
            # Calculate metrics for this entity type
            metrics = self._calculate_metrics(
                counts["tp"],
                counts["num_predicted"],
                counts["num_annotated"],
                beta,
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

    def _update_per_type_counts(
        self, annotation_spans, prediction_spans, per_type_metrics
    ):
        """
        Update the per-entity type counts for annotations and predictions.

        Args:
            annotation_spans: List of annotation Span objects
            prediction_spans: List of prediction Span objects
            per_type_metrics: Dictionary to track per-type metrics

        Returns:
            Dictionary with updated per-type counts
        """
        updated_metrics = per_type_metrics.copy()
        for ann_span in annotation_spans:
            updated_metrics[ann_span.entity_type]["num_annotated"] += 1
        for pred_span in prediction_spans:
            updated_metrics[pred_span.entity_type]["num_predicted"] += 1
        return updated_metrics

    def calculate_score(
        self,
        evaluation_results: List[EvaluationResult],
        entities: Optional[List[str]] = None,
        beta: float = 2.0,
    ) -> EvaluationResult:

        #TODO: Add docstring

        #TODO: filter evaluation_results by entities
        df = self.get_results_dataframe(evaluation_results)
        return self.calculate_score_on_df(df, beta=beta)

    def calculate_score_on_df(self, results_df: pd.DataFrame, beta: float = 2) -> SpanEvaluationResult:
        """
        Evaluate the predictions against ground truth annotations.

        This method orchestrates the evaluation process by:
        1. Initializing evaluation metrics
        2. Processing each sentence to get annotation and prediction spans
        3. Matching predictions with annotations and tracking metrics
        4. Creating and returning the final evaluation result

        :param results_df: DataFrame containing sentence_id, tokens, token start indices, annotations and predictions
        :param beta: The beta parameter for F-beta score calculation. Default is 2.

        Returns:
            SpanEvaluationResult object containing precision, recall, f_beta and per-type metrics
        """
        # Initialize metrics tracking structures
        metrics = self._initialize_metrics()

        # Process each sentence
        for _, sentence_df in results_df.groupby("sentence_id"):
            # Get and process spans for the sentence
            annotation_spans, prediction_spans = self._process_sentence_spans(
                sentence_df
            )

            # Update total counts
            metrics["total_num_annotated"] += len(annotation_spans)
            metrics["total_num_predicted"] += len(prediction_spans)

            # Update per-entity type counts
            self._update_per_type_counts(
                annotation_spans, prediction_spans, metrics["per_type_metrics"]
            )

            # Match predictions with annotations and update metrics
            self._match_predictions_with_annotations(
                annotation_spans, prediction_spans, metrics
            )

        # Create and return the final evaluation result
        return self._create_evaluation_result(metrics, beta)

    def _create_spans(self, df: pd.DataFrame, column: str) -> List[Span]:
        """
        Create spans from a DataFrame column.

        Args:
            df (pd.DataFrame): DataFrame containing the spans.
            column (str): Name of the column to extract spans from.

        Returns:
            List[Span]: List of Span objects created from the DataFrame.
        """

        spans = []
        current_entity_type = None
        current_tokens = []
        current_start = None
        current_token_start = None  # Add token position tracking
        curr_char_position = 0
        token_position = 0  # Add token position counter

        for idx, row in df.iterrows():
            entity_type = row[column]
            token = row["token"]
            token_length = len(token)
            # If this isn't the first token, add space before it
            if idx > df.index[0]:
                curr_char_position += 1  # Account for space between tokens

            token_start = curr_char_position
            token_end = curr_char_position + token_length

            if entity_type == "O":
                if current_entity_type and current_tokens:
                    normalized_tokens = self._normalize_tokens(current_tokens)
                    if normalized_tokens:
                        spans.append(
                            Span(
                                entity_type=current_entity_type,
                                entity_value=current_tokens,
                                start_position=current_start,
                                end_position=token_start - 2,
                                normalized_value=normalized_tokens,
                                token_start=current_token_start,
                                token_end=idx,
                            )
                        )
                    current_entity_type = None
                    current_tokens = []
                    current_start = None
                    current_token_start = None
                curr_char_position = token_end
                token_position += 1  # Increment token position
                continue

            if entity_type != current_entity_type:
                if current_entity_type and current_tokens:
                    normalized_tokens = self._normalize_tokens(current_tokens)
                    if normalized_tokens:
                        spans.append(
                            Span(
                                entity_type=current_entity_type,
                                entity_value=current_tokens,
                                start_position=current_start,
                                end_position=token_start - 2,
                                normalized_value=normalized_tokens,
                                token_start=current_token_start,
                                token_end=idx,
                            )
                        )
                current_entity_type = entity_type
                current_tokens = [token]
                current_start = token_start
                current_token_start = idx  # Set token start position

            else:
                current_tokens.append(token)
            curr_char_position = token_end
            token_position += 1  # Increment token position

        # Handle final span
        if current_entity_type and current_tokens:
            normalized_tokens = self._normalize_tokens(current_tokens)
            if normalized_tokens:
                spans.append(
                    Span(
                        entity_type=current_entity_type,
                        entity_value=current_tokens,
                        start_position=current_start,
                        end_position=curr_char_position,
                        normalized_value=normalized_tokens,
                        token_start=current_token_start,
                        token_end=df.index[-1] + 1
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
        precision = true_positives / num_predicted if num_predicted > 0 else 0.0
        recall = true_positives / num_annotated if num_annotated > 0 else 0.0
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
            annotation_spans, prediction_spans = self._process_sentence_spans(
                sentence_df
            )
            for ann in annotation_spans:
                for pred in prediction_spans:
                    iou = self.calculate_iou(ann, pred, self.char_based)
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
