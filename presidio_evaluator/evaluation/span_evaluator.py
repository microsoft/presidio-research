from typing import List, Dict, Optional
import pandas as pd
from collections import defaultdict
import string
from presidio_evaluator.evaluation.skipwords import get_skip_words
from presidio_evaluator.data_objects import Span


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
    ):
        """
        Initialize the SpanEvaluator for evaluating pii entities detection results.

        :param iou_threshold: Minimum Intersection over Union (IoU) threshold for considering spans as matching.
                            Value between 0 and 1, where higher values require more overlap (default: 0.5)
        :param skip_words: Optional list of custom skip words to ignore during token normalization.
                         If None, uses skip words from skipwords.py (default: None).
                         Pass an empty list ([]) to disable skip word removal entirely.
        :param schema: The labeling schema to use for span creation. Valid values:
                      - 'BIO': Use Begin/Inside/Outside labeling scheme
                      - None: Use default scheme with entity start indicators (default: None)
        :param beta: The beta parameter for F-beta score calculation. Default is 2.
        """
        self.iou_threshold = iou_threshold
        self.schema = schema
        self.punctuation = set(string.punctuation)
        self.beta = beta
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
            # Skip if token is just punctuation
            if all(c in self.punctuation for c in token):
                continue
            # Skip if token is in skip words list
            if token in self.skip_words:
                continue
            normalized.append(token)
        return normalized

    def _merge_adjacent_spans(self, spans: List[Span], df: pd.DataFrame) -> List[Span]:
        """
        Merge adjacent spans of the same entity type if separated only by punctuation or whitespace.

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
                # Use row slicing instead of filtering on df["start"]
                tokens = df.loc[
                    current.start_position : next_span.end_position - 1, "token"
                ].tolist()
                current = Span(
                    entity_type=current.entity_type,
                    entity_value=tokens,
                    start_position=current.start_position,
                    end_position=next_span.end_position,
                )
            else:
                merged.append(current)
                current = next_span
        merged.append(current)
        return merged

    def _are_spans_adjacent(self, span1: Span, span2: Span, df: pd.DataFrame) -> bool:
        """
        Check if two spans are adjacent, i.e., separated only by punctuation or whitespace tokens.

        :param span1: First Span object
        :param span2: Second Span object
        :param df: DataFrame containing the tokens
        :return: True if spans are adjacent, False otherwise
        """
        # Slice tokens between span1 and span2 using the row indices
        between_tokens = df.loc[
            span1.end_position : span2.start_position - 1, "token"
        ].tolist()
        return all(
            all(c in self.punctuation or c.isspace() for c in token)
            for token in between_tokens
        )

    @staticmethod
    def calculate_iou(span1: Span, span2: Span) -> float:
        """
        Calculate the Intersection over Union (IoU) between two spans using normalized tokens.

        :param span1: First Span object
        :param span2: Second Span object
        :param df: DataFrame containing the tokens
        :return: IoU score (float between 0 and 1)
        """
        print(f"span1: {span1.normalized_value}, span2: {span2.normalized_value}")
        set1 = set(span1.normalized_value)
        set2 = set(span2.normalized_value)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0
    
    def evaluate(self, results_df: pd.DataFrame) -> Dict:
        # Group by sentence
        sentences = results_df.groupby("sentence_id")

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        total_num_annotated = 0
        total_num_predicted = 0

        # Track metrics per entity type
        per_type_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "num_annotated": 0, "num_predicted": 0})

        # Error analysis tracking
        error_analysis = defaultdict(int)

        for _, sentence_df in sentences:
            # Create annotation spans
            annotation_spans = self._create_spans(sentence_df, "annotation")
            prediction_spans = self._create_spans(sentence_df, "prediction")

            # Merge adjacent spans
            annotation_spans = self._merge_adjacent_spans(annotation_spans, sentence_df)
            prediction_spans = self._merge_adjacent_spans(prediction_spans, sentence_df)

            total_num_annotated += len(annotation_spans)
            total_num_predicted += len(prediction_spans)

            # Count per-entity annotated and predicted
            for ann_span in annotation_spans:
                per_type_metrics[ann_span.entity_type]["num_annotated"] += 1
            for pred_span in prediction_spans:
                per_type_metrics[pred_span.entity_type]["num_predicted"] += 1

            # Track matched spans to avoid double-counting
            matched_preds = set()

            # For each annotation, find best matching prediction
            for ann_span in annotation_spans:
                best_match = None
                best_iou = 0.0

                for pred_span in prediction_spans:
                    pred_span_key = (
                        pred_span.entity_type,
                        pred_span.start_position,
                        pred_span.end_position,
                    )
                    if pred_span_key in matched_preds:
                        continue

                    # Only compare spans of same type
                    if pred_span.entity_type != ann_span.entity_type:
                        continue

                    iou = self.calculate_iou(ann_span, pred_span)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = pred_span

                entity_type = ann_span.entity_type

                # If match found above threshold
                if best_match and best_iou >= self.iou_threshold:
                    total_true_positives += 1
                    per_type_metrics[entity_type]["tp"] += 1
                    matched_preds.add(
                        (
                            best_match.entity_type,
                            best_match.start_position,
                            best_match.end_position,
                        )
                    )
                else:
                    total_false_negatives += 1
                    per_type_metrics[entity_type]["fn"] += 1
                    if best_match:
                        error_analysis[f"low_iou_{entity_type}"] += 1
                    else:
                        error_analysis[f"missed_{entity_type}"] += 1

            # Count false positives (unmatched predictions)
            for pred_span in prediction_spans:
                pred_span_key = (
                    pred_span.entity_type,
                    pred_span.start_position,
                    pred_span.end_position,
                )
                if pred_span_key not in matched_preds:
                    total_false_positives += 1
                    per_type_metrics[pred_span.entity_type]["fp"] += 1
                    error_analysis[f"extra_{pred_span.entity_type}"] += 1

        # Calculate global metrics using _calculate_metrics
        global_metrics = self._calculate_metrics(
            total_true_positives, total_num_predicted, total_num_annotated, self.beta
        )
        precision = global_metrics["precision"]
        recall = global_metrics["recall"]
        f_beta = global_metrics["f_beta"]

        # Calculate per-type metrics using the same logic as global
        per_type = {}
        for entity_type, counts in per_type_metrics.items():
            per_type[entity_type] = self._calculate_metrics(
                counts["tp"], counts["num_predicted"], counts["num_annotated"], self.beta
            )

        return {
            "precision": precision,
            "recall": recall,
            "f_beta": f_beta,
            "per_type": per_type,
        }
        

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
            is_entity_start = row["is_entity_start"]

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
            annotation_spans = self._create_spans(sentence_df, "annotation")
            prediction_spans = self._create_spans(sentence_df, "prediction")
            annotation_spans = self._merge_adjacent_spans(annotation_spans, sentence_df)
            prediction_spans = self._merge_adjacent_spans(prediction_spans, sentence_df)
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
