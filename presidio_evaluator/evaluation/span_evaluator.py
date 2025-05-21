from typing import List, Dict
import pandas as pd
from collections import defaultdict
import string
from spacy.lang.en.stop_words import STOP_WORDS
from presidio_evaluator.data_objects import Span

class SpanEvaluator:
    """
    Evaluates PII detection using span-based fuzzy matching with token-level Intersection over Union (IoU).
    """

    def __init__(self, iou_threshold: float = 0.5, schema: str = None):
        """
        Initialize the SpanEvaluator.
        
        :param iou_threshold: Minimum IoU threshold for considering spans as matching (default: 0.5)
        :param schema: The labeling schema to use ('BIO' or None for default)
        """
        self.iou_threshold = iou_threshold
        self.schema = schema
        # Common stop words to ignore during token normalization
        self.stop_words = STOP_WORDS
        self.punctuation = set(string.punctuation)

    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Normalize tokens by converting to lowercase, removing stop words, and removing standalone punctuation.
        
        :param tokens: List of token strings to normalize
        :return: List of normalized tokens
        """
        normalized = []
        for token in tokens:
            token = token.lower()
            # Skip if token is just punctuation
            if all(c in self.punctuation for c in token):
                continue
            # Skip if token is a stop word
            if token in self.stop_words:
                continue
            normalized.append(token)
        return normalized

    def merge_adjacent_spans(self, spans: List[Span], df: pd.DataFrame) -> List[Span]:
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
            if (current.entity_type == next_span.entity_type 
                and self._are_spans_adjacent(current, next_span, df)):
                # Use row slicing instead of filtering on df["start"]
                tokens = df.loc[current.start_position: next_span.end_position - 1, "token"].tolist()
                current = Span(
                    entity_type=current.entity_type,
                    entity_value=tokens,
                    start_position=current.start_position,
                    end_position=next_span.end_position
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
        between_tokens = df.loc[span1.end_position: span2.start_position - 1, "token"].tolist()
        return all(
            all(c in self.punctuation or c.isspace() for c in token)
            for token in between_tokens
        )


    def calculate_iou(self, span1: Span, span2: Span, df: pd.DataFrame) -> float:
        """
        Calculate the Intersection over Union (IoU) between two spans based on their normalized tokens.
        
        :param span1: First Span object
        :param span2: Second Span object
        :param df: DataFrame containing the tokens
        :return: IoU score (float between 0 and 1)
        """
        tokens1 = df.loc[span1.start_position: span1.end_position - 1, "token"].tolist()
        tokens2 = df.loc[span2.start_position: span2.end_position - 1, "token"].tolist()

        tokens1 = self.normalize_tokens(tokens1)
        tokens2 = self.normalize_tokens(tokens2)
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

    def evaluate(self, results_df: pd.DataFrame) -> Dict:
        """
        Evaluate model predictions using span-based fuzzy matching.
        
        :param results_df: DataFrame from Evaluator.get_results_dataframe() containing:
            - sentence_id
            - token
            - annotation (ground truth entity)
            - prediction (predicted entity)
        :return: Dictionary containing evaluation metrics:
            - precision
            - recall
            - f1
            - per_type: dict of precision, recall, f1 per entity type
            - error_analysis: dict containing common error patterns
        """
        # Group by sentence
        sentences = results_df.groupby("sentence_id")
        
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        
        # Track metrics per entity type
        per_type_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        # Error analysis tracking
        error_analysis = defaultdict(int)
        
        for _, sentence_df in sentences:
            # Create annotation spans
            annotation_spans = self._create_spans(sentence_df, "annotation")
            prediction_spans = self._create_spans(sentence_df, "prediction")
            
            # Merge adjacent spans
            annotation_spans = self.merge_adjacent_spans(annotation_spans, sentence_df)
            prediction_spans = self.merge_adjacent_spans(prediction_spans, sentence_df)
            
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
                        pred_span.end_position
                    )
                    if pred_span_key in matched_preds:
                        continue

                    # Only compare spans of same type
                    if pred_span.entity_type != ann_span.entity_type:
                        continue

                    iou = self.calculate_iou(ann_span, pred_span, sentence_df)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = pred_span

                entity_type = ann_span.entity_type

                # If match found above threshold
                if best_match and best_iou >= self.iou_threshold:
                    total_true_positives += 1
                    per_type_metrics[entity_type]["tp"] += 1
                    matched_preds.add((
                        best_match.entity_type,
                        best_match.start_position,
                        best_match.end_position
                    ))
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
                pred_span.end_position
            )
            if pred_span_key not in matched_preds:
                total_false_positives += 1
                per_type_metrics[pred_span.entity_type]["fp"] += 1
                error_analysis[f"extra_{pred_span.entity_type}"] += 1
        
        # Calculate overall metrics
        metrics = self._calculate_metrics(
            total_true_positives,
            total_false_positives,
            total_false_negatives
        )
        
        # Calculate per-type metrics
        per_type = {}
        for entity_type, counts in per_type_metrics.items():
            per_type[entity_type] = self._calculate_metrics(
                counts["tp"], counts["fp"], counts["fn"]
            )
        
        return {
            **metrics,
            "per_type": per_type,
        }

    def _create_spans(self, df: pd.DataFrame, column: str) -> List[Span]:
        """
        Create Span objects from a DataFrame using either BIO labeling scheme or default scheme with boolean entity start indicators.
        
        :param df: DataFrame containing tokens and entity labels. For default scheme, requires 'is_entity_start' boolean column.
                  For BIO scheme, expects labels in format 'B-TYPE'/'I-TYPE'/'O'.
        :param column: Column name containing entity labels ('annotation' or 'prediction')
        :return: List of Span objects
        """
        if self.schema == "BIO":
            return self._create_spans_bio(df, column)
        
        spans = []
        current_type = None
        current_tokens = []
        current_start = None

        for idx, row in df.iterrows():
            entity_type = row[column]
            is_entity_start = row["is_entity_start"]

            if entity_type == "O":
                if current_type:
                    spans.append(
                        Span(
                            entity_type=current_type,
                            entity_value=current_tokens,
                            start_position=current_start,
                            end_position=idx
                        )
                    )
                    current_type = None
                    current_tokens = []
                    current_start = None
                continue

            if is_entity_start or current_type != entity_type:
                if current_type:
                    spans.append(
                        Span(
                            entity_type=current_type,
                            entity_value=current_tokens,
                            start_position=current_start,
                            end_position=idx
                        )
                    )
                current_type = entity_type
                current_tokens = [row["token"]]
                current_start = idx
            else:
                current_tokens.append(row["token"])

        if current_type:
            spans.append(
                Span(
                    entity_type=current_type,
                    entity_value=current_tokens,
                    start_position=current_start,
                    end_position=df.index[-1] + 1
                )
            )

        return spans


    def _calculate_metrics(
        self, true_positives: int, false_positives: int, false_negatives: int
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        :param true_positives: Number of true positives
        :param false_positives: Number of false positives
        :param false_negatives: Number of false negatives
        :return: Dictionary containing precision, recall, and F1 metrics
        """
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
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
            annotation_spans = self.merge_adjacent_spans(annotation_spans, sentence_df)
            prediction_spans = self.merge_adjacent_spans(prediction_spans, sentence_df)
            for ann in annotation_spans:
                for pred in prediction_spans:
                    iou = self.calculate_iou(ann, pred, sentence_df)
                    records.append({
                        "sentence_id": sentence_id,
                        "annotation_span": ann,
                        "prediction_span": pred,
                        "ann_entity": ann.entity_type,
                        "ann_start": ann.start_position,
                        "ann_end": ann.end_position,
                        "pred_entity": pred.entity_type,
                        "pred_start": pred.start_position,
                        "pred_end": pred.end_position,
                        "iou": iou
                    })
        return pd.DataFrame(records)
    
    def _create_spans_bio(self, df: pd.DataFrame, column: str) -> List[Span]:
        """
        Create Span objects from a DataFrame using BIO labeling scheme.
        
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
                    spans.append(
                        Span(
                            entity_type=current_type,
                            entity_value=current_tokens,
                            start_position=current_start,
                            end_position=idx
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
                    spans.append(
                        Span(
                            entity_type=current_type,
                            entity_value=current_tokens,
                            start_position=current_start,
                            end_position=idx
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
            spans.append(
                Span(
                    entity_type=current_type,
                    entity_value=current_tokens,
                    start_position=current_start,
                    end_position=df.index[-1] + 1
                )
            )
        
        return spans
