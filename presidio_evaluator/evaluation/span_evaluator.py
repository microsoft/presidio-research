from typing import List, Optional, Union, Set, Tuple
import pandas as pd

from presidio_analyzer import AnalyzerEngine

from presidio_evaluator.evaluation import BaseEvaluator, ModelError, ErrorType
from presidio_evaluator.data_objects import Span
from presidio_evaluator.evaluation.evaluation_result import (
    EvaluationResult,
)
from presidio_evaluator.models import BaseModel


class SpanEvaluator(BaseEvaluator):
    """
    Evaluates PII detection using span-based fuzzy matching with character-level Intersection over Union (IoU).
    """

    def __init__(
        self,
        model: Union[BaseModel, AnalyzerEngine],
        verbose: bool = False,
        compare_by_io: bool = True,
        entities_to_keep: Optional[List[str]] = None,
        generic_entities: Optional[List[str]] = None,
        skip_words: Optional[List] = None,
        iou_threshold: float = 0.9,
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
        self.char_based = char_based

    def _normalize_tokens(
        self, tokens: List[str], start_indices: Optional[List[int]] = None
    ) -> Union[List[str], Tuple[List[str], List[int]]]:
        """
        Normalize tokens by:
        1. Converting to lowercase
        2. Removing stop words
        3. Removing standalone punctuation
        4. Removing skip words (common words that shouldn't affect entity matching)

        :param tokens: List of token strings to normalize
        :return: List of normalized tokens
        """

        if not start_indices:
            start_indices = [None] * len(tokens)  # placeholder
        normalized = []
        normalized_indices = []
        for token, start in zip(tokens, start_indices):
            token = token.lower()
            # Skip if token is in skip words list
            if token in self.skip_words:
                continue
            normalized.append(token)
            normalized_indices.append(start)

        if not start_indices:
            return normalized

        return normalized, normalized_indices

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
                merged_tokens = [current.entity_value, next_span.entity_value]
                merged_normalized_text = (
                    current.normalized_tokens + next_span.normalized_tokens
                )
                current = Span(
                    entity_type=current.entity_type,
                    entity_value=" ".join(merged_tokens),
                    start_position=current.start_position,
                    end_position=next_span.end_position,
                    normalized_start_index=min(
                        current.normalized_start_index, next_span.normalized_start_index
                    ),
                    normalized_end_index=max(
                        current.normalized_end_index, next_span.normalized_end_index
                    ),
                    normalized_tokens=merged_normalized_text,
                    token_start=current.token_start,
                    token_end=next_span.token_end,
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
        non_skip_tokens = [
            tok for tok in between_tokens if tok.lower().strip() not in self.skip_words
        ]
        return len(non_skip_tokens) == 0

    @staticmethod
    def calculate_iou(
        span1: Span,
        span2: Span,
        ignore_entity_type: bool = True,
        use_normalized_indices: bool = True,
        char_based: bool = True,
    ) -> float:
        """
        Calculate the Intersection over Union (IoU) between two spans at character or token level.

        :param span1: First Span object
        :param span2: Second Span object
        :param ignore_entity_type: If True, ignores the entity type when calculating IoU
        :param use_normalized_indices: If True, uses normalized indices for IoU calculation
        :param char_based: If True, calculates IoU at character level, else at token level

        """
        if char_based:
            iou = span1.iou(
                other=span2,
                ignore_entity_type=ignore_entity_type,
                use_normalized_indices=use_normalized_indices,
            )
        else:
            range1 = set(span1.normalized_tokens)
            range2 = set(span2.normalized_tokens)

            intersection = len(range1.intersection(range2))
            union = len(range1.union(range2))

            iou = intersection / union if union > 0 else 0.0

        return iou

    def _process_sentence_spans(
        self, sentence_df: pd.DataFrame
    ) -> Tuple[List[Span], List[Span]]:
        annotation_spans = self._create_spans(df=sentence_df, column="annotation")
        prediction_spans = self._create_spans(df=sentence_df, column="prediction")

        annotation_spans = self._merge_adjacent_spans(
            spans=annotation_spans, df=sentence_df
        )
        prediction_spans = self._merge_adjacent_spans(
            spans=prediction_spans, df=sentence_df
        )

        return annotation_spans, prediction_spans

    @staticmethod
    def _handle_unmatched_predictions(
        prediction_spans: List[Span],
        matched_preds: Set[Tuple[str, int, int]],
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """
        Handle predictions that weren't matched to any annotation.

        :param prediction_spans: List of prediction Span objects
        :param matched_preds: Set of already matched prediction spans
        :param evaluation_result: EvaluationResult object to update

        """
        if not evaluation_result.model_errors:
            evaluation_result.model_errors = []

        for pred_span in prediction_spans:
            pred_span_key = (
                pred_span.entity_type,
                pred_span.start_position,
                pred_span.end_position,
            )
            if pred_span_key not in matched_preds:
                evaluation_result.results[("O", pred_span.entity_type)] += 1
                evaluation_result.pii_false_positives += 1
                evaluation_result.per_type[pred_span.entity_type].false_positives += 1
                model_error = ModelError(
                    error_type=ErrorType.FP,
                    annotation="O",
                    prediction=pred_span.entity_type,
                    full_text=pred_span.entity_value,
                    token=" ".join(pred_span.normalized_tokens),
                    explanation=f"False positive for {pred_span}",
                )
                evaluation_result.model_errors.append(model_error)

        return evaluation_result

    @staticmethod
    def _check_if_matched_already(
        pred_span: Span, ann_span: Span, matched_preds: set
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

        return True

    def _find_best_match(
        self,
        ann_span: Span,
        prediction_spans: List[Span],
        matched_preds: Set[Tuple[str, int, int]],
    ) -> Tuple[Optional[Span], float]:
        """
        Find the best matching prediction span for a given annotation span.

        :param ann_span: The annotation Span to match against
        :param prediction_spans: List of prediction Span objects
        :param matched_preds: Set of already matched prediction spans to avoid duplicates
        """
        best_match = None
        best_iou = 0.0

        for pred_span in prediction_spans:
            if self._check_if_matched_already(
                pred_span=pred_span, ann_span=ann_span, matched_preds=matched_preds
            ):
                iou = self.calculate_iou(
                    span1=ann_span,
                    span2=pred_span,
                    ignore_entity_type=True,
                    use_normalized_indices=True,
                    char_based=self.char_based,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_span

        return best_match, best_iou

    def _create_evaluation_result(
        self,
        evaluation_result: EvaluationResult,
        true_positives: int,
        num_predicted: int,
        num_annotated: int,
        beta: float,
    ) -> EvaluationResult:
        """
        Update the evaluation result with overall metrics and per-type metrics.

        :param evaluation_result: EvaluationResult object to update
        :param true_positives: Total number of true positive predictions
        :param num_predicted: Total number of predicted spans
        :param num_annotated: Total number of annotated spans
        :param beta: The beta parameter for F-beta score calculation.
        """

        precision, recall, f_beta = self._calculate_metrics(
            true_positives, num_predicted, num_annotated, beta
        )
        evaluation_result.pii_recall = recall
        evaluation_result.pii_precision = precision
        evaluation_result.pii_f = f_beta
        evaluation_result = self._calculate_per_type_metrics(evaluation_result, beta)
        return evaluation_result

    def _calculate_per_type_metrics(
        self,
        evaluation_result: EvaluationResult,
        beta: float,
    ) -> EvaluationResult:
        """
        Update per-type metrics in the evaluation result.

        :param evaluation_result: EvaluationResult object containing per-type metrics
        :param beta: F-beta parameter

        """

        for entity_type, pii_metrics in evaluation_result.per_type.items():
            # Calculate metrics for this entity type
            precision, recall, f_beta = self._calculate_metrics(
                pii_metrics.true_positives,
                pii_metrics.num_predicted,
                pii_metrics.num_annotated,
                beta,
            )
            pii_metrics.precision = precision
            pii_metrics.recall = recall
            pii_metrics.f_beta = f_beta
        return evaluation_result

    @staticmethod
    def create_global_entities_df(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a DataFrame containing global PII entities from the results DataFrame.

        :param results_df: DataFrame containing the evaluation results
        :return: DataFrame with global entities and their counts
        """
        # Create a deep copy to avoid modifying the original DataFrame
        global_df = results_df.copy(deep=True)
        global_df["annotation"] = global_df["annotation"].apply(
            lambda x: "O" if x == "O" else "PII"
        )
        global_df["prediction"] = global_df["prediction"].apply(
            lambda x: "O" if x == "O" else "PII"
        )
        return global_df

    def calculate_score(
        self,
        evaluation_results: List[EvaluationResult],
        entities: Optional[List[str]] = None,
        beta: float = 2.0,
    ) -> EvaluationResult:
        """
        Calculate the evaluation score based on the provided evaluation results (evaluation run).
        :param evaluation_results: List of EvaluationResult objects containing the results of the evaluation run,
        specifically `actual_tags` and `predicted_tags`.
        :param entities: Optional list of entities to filter the evaluation results by.
        If None, all entities are considered.
        :param beta: The beta parameter for F-beta score calculation. Default is 2.
        """
        evaluation_result = EvaluationResult()
        df = self.get_results_dataframe(
            evaluation_results=evaluation_results, entities=entities
        )
        evaluation_result = self.calculate_score_on_df(
            per_type=True, results_df=df, beta=beta, evaluation_result=evaluation_result
        )
        global_pii_df = self.create_global_entities_df(results_df=df)
        evaluation_result = self.calculate_score_on_df(
            per_type=False,
            results_df=global_pii_df,
            beta=beta,
            evaluation_result=evaluation_result,
        )
        return evaluation_result

    def calculate_score_on_df(
        self,
        per_type: bool,
        results_df: pd.DataFrame,
        beta: float = 2,
        evaluation_result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate the predictions against ground truth annotations.
        This method processes a DataFrame containing evaluation results and calculates metrics
        using span-based fuzzy matching with IoU threshold. It can operate in two modes:
        per-entity type evaluation or global PII evaluation.

        :param per_type: If True, performs per-entity type evaluation; if False, performs
                    global PII vs non-PII evaluation
        :param results_df: DataFrame containing sentence_id, tokens, token start indices,
                        annotations and predictions columns
        :param beta: The beta parameter for F-beta score calculation. Higher values weight
                    recall more than precision. Default is 2.
        :param evaluation_result: Optional existing EvaluationResult to update. If None,
                                creates a new one.
        :return: EvaluationResult object containing computed metrics, counts, and error analysis

        """
        if not evaluation_result:
            evaluation_result = EvaluationResult()

        # Process each sentence
        for _, sentence_df in results_df.groupby("sentence_id"):
            # Get and process spans for the sentence
            annotation_spans, prediction_spans = self._process_sentence_spans(
                sentence_df
            )
            # Match predictions with annotations and update metrics
            evaluation_result = self._match_predictions_with_annotations(
                annotation_spans, prediction_spans, evaluation_result, per_type
            )
        # Create and return the final evaluation result
        return self._create_evaluation_result(
            evaluation_result,
            evaluation_result.pii_true_positives,
            evaluation_result.pii_predicted,
            evaluation_result.pii_annotated,
            beta,
        )

    def _create_spans(self, df: pd.DataFrame, column: str) -> List[Span]:
        """
        Create spans from a DataFrame column.

        :param df: DataFrame containing the spans.
        :param column: Name of the column to extract spans from.

        Returns:
            List[Span]: List of Span objects created from the DataFrame.
        """
        spans = []
        current_entity_type = None
        current_tokens = []
        current_start_indices = []
        current_token_start = None  # Add token position tracking
        curr_char_position = 0
        token_position = 0  # Add token position counter

        for idx, (_, row) in enumerate(df.iterrows()):
            entity_type = row[column]
            token = row["token"]
            token_start = row["start_indices"]
            token_length = len(token)
            # If this isn't the first token, add space before it
            if idx > df.index[0]:
                curr_char_position += 1  # Account for space between tokens

            token_end = curr_char_position + token_length

            if entity_type == "O":
                if current_entity_type and current_tokens:
                    normalized_tokens, normalized_start_indices = (
                        self._normalize_tokens(current_tokens, current_start_indices)
                    )
                    if normalized_tokens:
                        spans.append(
                            self.__create_span(
                                entity_type=current_entity_type,
                                start_indices=current_start_indices,
                                token_start=current_token_start,
                                current_tokens=current_tokens,
                                idx=idx,
                                normalized_satrt_indices=normalized_start_indices,
                                normalized_tokens=normalized_tokens,
                            )
                        )
                    current_entity_type = None
                    current_tokens = []
                    current_start_indices = []
                    current_token_start = None

                curr_char_position = token_end
                token_position += 1  # Increment token position
                continue

            if entity_type != current_entity_type:
                if current_entity_type and current_tokens:
                    normalized_tokens, normalized_start_indices = (
                        self._normalize_tokens(current_tokens, current_start_indices)
                    )
                    if normalized_tokens:
                        spans.append(
                            self.__create_span(
                                entity_type=current_entity_type,
                                start_indices=current_start_indices,
                                token_start=current_token_start,
                                current_tokens=current_tokens,
                                idx=idx,
                                normalized_satrt_indices=normalized_start_indices,
                                normalized_tokens=normalized_tokens,
                            )
                        )
                current_entity_type = entity_type
                current_tokens = [token]
                current_start_indices = [token_start]
                current_token_start = idx  # Set token start position

            else:
                current_tokens.append(token)
                current_start_indices.append(token_start)
            curr_char_position = token_end
            token_position += 1  # Increment token position

        # Handle final span
        if current_entity_type and current_tokens:
            normalized_tokens, normalized_start_indices = self._normalize_tokens(
                current_tokens, current_start_indices
            )
            if normalized_tokens:
                spans.append(
                    self.__create_span(
                        entity_type=current_entity_type,
                        start_indices=current_start_indices,
                        token_start=current_token_start,
                        current_tokens=current_tokens,
                        idx=df.index[-1] + 1,
                        normalized_satrt_indices=normalized_start_indices,
                        normalized_tokens=normalized_tokens,
                    )
                )
        return spans

    def __create_span(
        self,
        entity_type: str,
        start_indices: List[int],
        token_start: int,
        current_tokens: List[str],
        idx: int,
        normalized_satrt_indices: List[int],
        normalized_tokens: List[str],
    ):
        return Span(
            entity_type=entity_type,
            entity_value=" ".join(current_tokens),
            start_position=start_indices[0],
            end_position=start_indices[-1] + len(current_tokens[-1]),
            normalized_tokens=normalized_tokens,
            normalized_start_index=min(normalized_satrt_indices),
            normalized_end_index=self._get_normalized_end_index(
                normalized_tokens, normalized_satrt_indices
            ),
            token_start=token_start,
            token_end=idx,
        )

    @staticmethod
    def _get_normalized_end_index(
        normalized_tokens: List[str], normalized_indices: List[int]
    ) -> int:
        """Calculate the end character index of the last token in the normalized tokens list."""  # noqa: E501
        return max(
            [
                len(tok) + start
                for tok, start in zip(normalized_tokens, normalized_indices)
            ]
        )

    def _calculate_metrics(
        self,
        true_positives: int,
        num_predicted: int,
        num_annotated: int,
        beta: float = 2,
    ) -> tuple[float, float, float]:
        """Calculate precision, recall, and F-beta score using the new logic.

        :param true_positives: Number of true positives
        :param num_predicted: Number of predicted spans
        :param num_annotated: Number of annotated (gold) spans
        :param beta: The beta parameter for F-beta score calculation. Default is 2.
        :return: Dictionary containing precision, recall, and f-beta metrics
        """
        precision = self.precision(tp=true_positives, num_predicted=num_predicted)
        recall = self.recall(tp=true_positives, num_annotated=num_annotated)
        f_beta = self.f_beta(precision=precision, recall=recall, beta=beta)
        return precision, recall, f_beta

    
    
    def _match_predictions_with_annotations(
        self,
        annotation_spans: List[Span],
        prediction_spans: List[Span],
        evaluation_result: EvaluationResult,
        per_type: bool = True,
    ) -> EvaluationResult:
        
        if not evaluation_result.model_errors:
            evaluation_result.model_errors = []
        
        for ann_span in annotation_spans:
            overlapping_preds = []
            for pred_span in prediction_spans:
                iou = iou = self.calculate_iou(
                    ann_span, pred_span, char_based=self.char_based
                )
                if iou > 0:
                    overlapping_preds.append((pred_span, iou))                    
            
            overlapping_preds.sort(key=lambda x: x[0].start_position)

            if not overlapping_preds:
                if per_type:
                    evaluation_result.per_type[
                        ann_span.entity_type
                    ].false_negatives += 1
                    
                else:
                    evaluation_result.pii_false_negatives += 1
                evaluation_result.results[
                            (ann_span.entity_type, "O")
                        ] = (
                            evaluation_result.results.get(
                                (ann_span.entity_type, "O"), 0
                            )
                            + 1
                        )
                evaluation_result.model_errors.append(
                        ModelError(
                            error_type=ErrorType.FN,
                            annotation=ann_span.entity_type,
                            prediction="O",
                            full_text=ann_span.entity_value,
                            token=" ".join(ann_span.normalized_tokens),
                            explanation=f"Entity {ann_span.entity_type} not detected: {ann_span.entity_value}",
                        )
                    )
            elif len(overlapping_preds) == 1:
                if overlapping_preds[0][1] >= self.iou_threshold:
                    pred_span = overlapping_preds[0][0]
                    if pred_span.entity_type == ann_span.entity_type:
                        if per_type:
                            evaluation_result.per_type[
                                ann_span.entity_type
                            ].true_positives += 1
                            
                        else:
                            evaluation_result.pii_true_positives += 1
                    else:
                        if per_type:
                            evaluation_result.per_type[
                                ann_span.entity_type
                            ].false_negatives += 1
                        else:
                            evaluation_result.pii_false_negatives += 1
                            evaluation_result.pii_false_positives += 1
                        evaluation_result.results[
                            (ann_span.entity_type, pred_span.entity_type)
                        ] = (
                            evaluation_result.results.get(
                                (ann_span.entity_type, pred_span.entity_type), 0
                            )
                            + 1
                        )
                        evaluation_result.model_errors.append(
                        ModelError(
                            error_type=ErrorType.FN,
                            annotation=ann_span.entity_type,
                            prediction=pred_span.entity_type,
                            full_text=ann_span.entity_value,
                            token=" ".join(ann_span.normalized_tokens),
                            explanation=f"Entity {ann_span.entity_type} not detected: {ann_span.entity_value}",
                        )
                    )
                        evaluation_result.model_errors.append(
                        ModelError(
                            error_type=ErrorType.FP,
                            annotation=ann_span.entity_type,
                            prediction=pred_span.entity_type,
                            full_text=pred_span.entity_value,
                            token=" ".join(pred_span.normalized_tokens),
                            explanation=f"False prediction: {pred_span.entity_type}",
                        )
                    )
                        evaluation_result.model_errors.append(
                        ModelError(
                            error_type=ErrorType.WrongEntity,
                            annotation=ann_span.entity_type,
                            prediction=pred_span.entity_type,
                            full_text=ann_span.entity_value,
                            token=" ".join(ann_span.normalized_tokens),
                            explanation=f"Wrong entity type: {ann_span.entity_type} detected as {pred_span.entity_type}, iou={iou:.2f}",
                        )
                    )

                else:
                    if ann_span.entity_type == pred_span.entity_type:
                        if per_type:
                            evaluation_result.per_type[
                                ann_span.entity_type
                            ].false_negatives += 1
                        else:
                            evaluation_result.pii_false_negatives += 1
                        evaluation_result.results[
                            (ann_span.entity_type, "O")
                        ] = (
                            evaluation_result.results.get(
                                (ann_span.entity_type, "O"), 0
                            )
                            + 1
                        )
                        evaluation_result.model_errors.append(
                        ModelError(
                            error_type=ErrorType.FN,
                            annotation=ann_span.entity_type,
                            prediction="O",
                            full_text=ann_span.entity_value,
                            token=" ".join(ann_span.normalized_tokens),
                            explanation=f"Same entity {ann_span.entity_type} with low iou, iou={iou:.2f}",
                        )
                    )
                    else:
                        if per_type:
                            evaluation_result.per_type[
                                ann_span.entity_type
                            ].false_negatives += 1
                            evaluation_result.per_type[
                                ann_span.entity_type
                            ].false_positives += 1
                        else:
                            evaluation_result.pii_false_negatives += 1
                            evaluation_result.pii_false_positives += 1
                        
                        evaluation_result.model_errors.append(
                        ModelError(
                            error_type=ErrorType.FN,
                            annotation=ann_span.entity_type,
                            prediction="O",
                            full_text=ann_span.entity_value,
                            token=" ".join(ann_span.normalized_tokens),
                            explanation=f"Entity {ann_span.entity_type} not detected: {ann_span.entity_value}, iou={iou:.2f}",
                        )
                    )
                        evaluation_result.model_errors.append(
                        ModelError(
                            error_type=ErrorType.FP,
                            annotation="O",
                            prediction=pred_span.entity_type,
                            full_text=pred_span.entity_value,
                            token=" ".join(pred_span.normalized_tokens),
                            explanation=f"False prediction: {pred_span.entity_type}",
                        )
                    )
                        evaluation_result.results[
                            (ann_span.entity_type, "O")
                        ] = (
                            evaluation_result.results.get(
                                (ann_span.entity_type, "O"), 0
                            )
                            + 1
                        )
                        evaluation_result.results[
                            ("O", pred_span.entity_type)
                        ] = (
                            evaluation_result.results.get(
                                ("O", pred_span.entity_type), 0
                            )
                            + 1
                        )
            else:
                cumulative_preds = []
                current_iou = 0
                for pred_span, iou in overlapping_preds:
                    combined_iou = self._calculate_combined_iou(ann_span, cumulative_preds + [pred_span])

                    if combined_iou > current_iou:
                        cumulative_preds.append(pred_span)
                        current_iou = combined_iou

                
                if current_iou >= self.iou_threshold:
                    for pred_span in cumulative_preds:
                        if pred_span.entity_type == ann_span.entity_type:
                            if per_type:
                                evaluation_result.per_type[
                                    ann_span.entity_type
                                ].true_positives += 1
                                evaluation_result.per_type[
                                    ann_span.entity_type
                                ].num_predicted += len(cumulative_preds)
                            else:
                                evaluation_result.pii_true_positives += 1
                                evaluation_result.pii_predicted += len(cumulative_preds)
                            evaluation_result.results[
                                (ann_span.entity_type, pred_span.entity_type)
                            ] = (
                                evaluation_result.results.get(
                                    (ann_span.entity_type, pred_span.entity_type), 0
                                )
                                + 1
                            )
                        else:
                            if per_type:
                                evaluation_result.per_type[
                                    ann_span.entity_type
                                ].false_negatives += 1
                                evaluation_result.per_type[
                                    pred_span.entity_type
                                ].false_positives += 1
                            else:
                                evaluation_result.pii_false_negatives += 1
                                evaluation_result.pii_false_positives += 1
                            evaluation_result.results[
                                (ann_span.entity_type, "O")
                            ] = (
                                evaluation_result.results.get(
                                    (ann_span.entity_type, "O"), 0
                                )
                                + 1
                            )
                            evaluation_result.results[
                                ("O", pred_span.entity_type)
                            ] = (
                                evaluation_result.results.get(
                                    ("O", pred_span.entity_type), 0
                                )
                                + 1
                            )
                            evaluation_result.model_errors.append(
                            ModelError(
                                error_type=ErrorType.WrongEntity,
                                annotation=ann_span.entity_type,
                                prediction=pred_span.entity_type,
                                full_text=ann_span.entity_value,
                                token=" ".join(ann_span.normalized_tokens),
                                explanation=f"Wrong entity type: {ann_span.entity_type} detected as {pred_span.entity_type}, iou={iou:.2f}",
                            )
                        )
                            evaluation_result.model_errors.append(
                            ModelError(
                                error_type=ErrorType.FN,
                                annotation=ann_span.entity_type,
                                prediction=pred_span.entity_type,
                                full_text=ann_span.entity_value,
                                token=" ".join(ann_span.normalized_tokens),
                                explanation=f"Entity {ann_span.entity_type} not detected: {ann_span.entity_value}",
                            )
                        )
                            evaluation_result.model_errors.append(
                            ModelError(
                                error_type=ErrorType.FP,
                                annotation=ann_span.entity_type,
                                prediction=pred_span.entity_type,
                                full_text=pred_span.entity_value,
                                token=" ".join(pred_span.normalized_tokens),
                                explanation=f"False prediction: {pred_span.entity_type}",
                            )
                        )
                else:
                    if ann_span.entity_type == pred_span.entity_type:
                        if per_type:
                            evaluation_result.per_type[
                                ann_span.entity_type
                            ].false_negatives += 1
                        else:
                            evaluation_result.pii_false_negatives += 1
                        evaluation_result.results[
                            (ann_span.entity_type, "O")
                        ] = (
                            evaluation_result.results.get(
                                (ann_span.entity_type, "O"), 0
                            )
                            + 1
                        )
                        evaluation_result.model_errors.append(
                        ModelError(
                            error_type=ErrorType.FN,
                            annotation=ann_span.entity_type,
                            prediction="O",
                            full_text=ann_span.entity_value,
                            token=" ".join(ann_span.normalized_tokens),
                            explanation=f"Same entity {ann_span.entity_type} with low iou, iou={iou:.2f}",
                        )
                    )                 
        return evaluation_result        

                
                    
        
    
    # def _match_predictions_with_annotations(
    #     self,
    #     annotation_spans: List[Span],
    #     prediction_spans: List[Span],
    #     evaluation_result: EvaluationResult,
    #     per_type: bool = True,
    # ) -> EvaluationResult:
    #     """
    #     Match predictions to annotations in a single pass.
    #     For each annotation, incrementally add prediction spans as long as they improve the IoU.
    #     Only tracks per-entity type metrics.

    #     :param annotation_spans: List of annotation Span objects
    #     :param prediction_spans: List of prediction Span objects
    #     :param evaluation_result: EvaluationResult object to update
    #     :return: Updated EvaluationResult
    #     """
    #     matched_predictions = set()

    #     if not evaluation_result.model_errors:
    #         evaluation_result.model_errors = []

    #     # Process each annotation span
    #     for ann_span in annotation_spans:
    #         annotated_type = ann_span.entity_type
    #         self._add_to_annotated(evaluation_result, per_type, ann_span)
    #         # Find all intersecting prediction spans for per-entity metrics
    #         overlapping_preds = []
    #         for pred_span in prediction_spans:
    #             # Skip already matched predictions (for per-entity metrics)
    #             pred_key = (
    #                 pred_span.entity_type,
    #                 pred_span.start_position,
    #                 pred_span.end_position,
    #             )
    #             if pred_key in matched_predictions:
    #                 continue

    #             iou = self.calculate_iou(
    #                 ann_span, pred_span, char_based=self.char_based
    #             )
    #             if iou > 0:  # Any overlap
    #                 overlapping_preds.append((pred_span, iou))

    #         # Sort by position for consistent processing
    #         overlapping_preds.sort(key=lambda x: x[0].start_position)

    #         # For per-entity metrics (type-specific)
    #         matched_preds = []
    #         current_iou = 0.0

    #         # Only consider predictions with matching entity types for per-entity metrics
    #         type_matching_preds = [
    #             (p, iou)
    #             for p, iou in overlapping_preds
    #             if p.entity_type == ann_span.entity_type
    #             and (p.entity_type, p.start_position, p.end_position)
    #             not in matched_predictions
    #         ]

    #         evaluation_result = self._update_wrong_entities(
    #             overlapping_preds,
    #             annotated_type,
    #             matched_predictions,
    #             ann_span,
    #             evaluation_result,
    #         )

    #         for pred_span, _ in type_matching_preds:
    #             test_preds = matched_preds + [pred_span]
    #             combined_iou = self._calculate_combined_iou(ann_span, test_preds)

    #             if combined_iou > current_iou:
    #                 matched_preds.append(pred_span)
    #                 current_iou = combined_iou

    #         # For per-entity metrics, if we have matching entity type predictions with good IoU
    #         if current_iou >= self.iou_threshold and matched_preds:
    #             # Record true positives for this entity type
    #             if per_type:
    #                 evaluation_result.per_type[ann_span.entity_type].true_positives += 1
    #                 evaluation_result.per_type[ann_span.entity_type].num_predicted += 1
    #             else:
    #                 evaluation_result.pii_true_positives += 1
    #                 evaluation_result.pii_predicted += 1
    #             # Mark these predictions as matched for per-entity metrics
    #             for pred_span in matched_preds:
    #                 pred_key = (
    #                     pred_span.entity_type,
    #                     pred_span.start_position,
    #                     pred_span.end_position,
    #                 )
    #                 matched_predictions.add(pred_key)
    #                 # Record entity matches/mismatches in confusion matrix
    #                 evaluation_result.results[
    #                     (ann_span.entity_type, pred_span.entity_type)
    #                 ] = (
    #                     evaluation_result.results.get(
    #                         (ann_span.entity_type, pred_span.entity_type), 0
    #                     )
    #                     + 1
    #                 )
    #         else:
    #             # False negative for this entity type
    #             if per_type:
    #                 evaluation_result.per_type[
    #                     ann_span.entity_type
    #                 ].false_negatives += 1
    #             else:
    #                 evaluation_result.pii_false_negatives += 1
    #             evaluation_result.model_errors.append(
    #                 ModelError(
    #                     error_type=ErrorType.FN,
    #                     annotation=ann_span.entity_type,
    #                     prediction="O",
    #                     full_text=ann_span.entity_value,
    #                     token=" ".join(ann_span.normalized_tokens),
    #                     explanation=f"Entity {ann_span.entity_type} not detected: {ann_span.entity_value}",
    #                 )
    #             )

    #     # Process unmatched predictions (false positives) for per-entity metrics
    #     for pred_span in prediction_spans:
    #         pred_key = (
    #             pred_span.entity_type,
    #             pred_span.start_position,
    #             pred_span.end_position,
    #         )

    #         # Per-entity metrics - record false positives
    #         if pred_key not in matched_predictions:
    #             if per_type:
    #                 evaluation_result.per_type[
    #                     pred_span.entity_type
    #                 ].false_positives += 1
    #                 evaluation_result.per_type[pred_span.entity_type].num_predicted += 1
    #             else:
    #                 evaluation_result.pii_false_positives += 1
    #                 evaluation_result.pii_predicted += 1
    #             evaluation_result.model_errors.append(
    #                 ModelError(
    #                     error_type=ErrorType.FP,
    #                     annotation="O",
    #                     prediction=pred_span.entity_type,
    #                     full_text=pred_span.entity_value,
    #                     token=" ".join(pred_span.normalized_tokens),
    #                     explanation=f"False prediction: {pred_span.entity_type}",
    #                 )
    #             )

    #     return evaluation_result

    @staticmethod
    def _update_wrong_entities(
        overlapping_preds,
        annotated_entity_type,
        matched_predictions,
        ann_span,
        evaluation_result,
    ) -> EvaluationResult:
        non_type_matching_preds = [
            (p, iou)
            for p, iou in overlapping_preds
            if p.entity_type != annotated_entity_type
            and (p.entity_type, p.start_position, p.end_position)
            not in matched_predictions
        ]

        for non_matching_pred, iou in non_type_matching_preds:
            # Record entity type mismatches in error analysis
            if non_matching_pred.entity_type != ann_span.entity_type:
                evaluation_result.model_errors.append(
                    ModelError(
                        error_type=ErrorType.WrongEntity,
                        annotation=ann_span.entity_type,
                        prediction=non_matching_pred.entity_type,
                        full_text=ann_span.entity_value,
                        token=" ".join(ann_span.normalized_tokens),
                        explanation=f"Wrong entity type: {ann_span.entity_type} detected as {non_matching_pred.entity_type}, iou={iou:.2f}",
                    )
                )
            evaluation_result.results[
                (ann_span.entity_type, non_matching_pred.entity_type)
            ] = (
                evaluation_result.results.get(
                    (ann_span.entity_type, non_matching_pred.entity_type), 0
                )
                + 1
            )

        return evaluation_result

    def _add_to_annotated(self, evaluation_result, per_type, ann_span):
        if per_type:
            evaluation_result.per_type[ann_span.entity_type].num_annotated += 1
        else:
            evaluation_result.pii_annotated += 1

    def _calculate_combined_iou(
        self, annotation_span: Span, prediction_spans: List[Span]
    ) -> float:
        """
        Calculate the combined IoU of multiple prediction spans against an annotation span.

        :param annotation_span: The annotation span to match against
        :param prediction_spans: List of prediction spans that potentially overlap
        :return: Combined IoU value between 0 and 1
        """
        if not prediction_spans:
            return 0.0

        if self.char_based:
            # Character-based IoU
            ann_chars = set(
                range(
                    annotation_span.normalized_start_index,
                    annotation_span.normalized_end_index + 1,
                )
            )
            pred_chars = set()
            for i, pred_span in enumerate(prediction_spans):
                if i == 0:
                    pred_chars.update(
                        range(
                            pred_span.normalized_start_index,
                            pred_span.normalized_end_index + 1,
                        )
                    )
                else:
                    pred_chars.update(
                        range(
                            pred_span.normalized_start_index - 1,
                            pred_span.normalized_end_index,
                        )
                    )
            intersection = len(ann_chars.intersection(pred_chars))
            union = len(ann_chars.union(pred_chars))
        else:
            # Token-based IoU
            ann_tokens = set(annotation_span.normalized_tokens)
            pred_tokens = set()
            for pred_span in prediction_spans:
                pred_tokens.update(pred_span.normalized_tokens)

            intersection = len(ann_tokens.intersection(pred_tokens))
            union = len(ann_tokens.union(pred_tokens))
        return intersection / union if union > 0 else 0.0
