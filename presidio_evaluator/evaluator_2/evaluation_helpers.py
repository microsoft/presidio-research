from copy import deepcopy
from typing import Dict, List
from collections import Counter
import pandas as pd

from presidio_evaluator import InputSample
from presidio_evaluator.evaluator_2 import SpanOutput


def get_span_eval_schema(
        span_outputs: List[SpanOutput], entities_to_keep: List[str]
) -> Dict[str, Dict[str, Counter]]:
    """Update the evaluation schema with the span_outputs.
    param:span_outputs (dict): The new schema to update the evaluation schema with.
    returns: dict: The updated evaluation schema.
    """
    # set up a dict for storing the span metrics
    span_cat_output = \
        {"correct": 0, "incorrect": 0, "partial": 0, "missed": 0, "spurious": 0}
    # copy results dict to cover the four evaluation schemes for PII.
    span_pii_eval = {
        "strict": Counter(span_cat_output),
        "ent_type": Counter(span_cat_output),
        "partial": Counter(span_cat_output),
        "exact": Counter(span_cat_output),
    }
    # copy results dict to cover the four evaluation schemes
    # for each entity in entities_to_keep.
    span_ent_eval = {
        e: deepcopy(span_pii_eval) for e in entities_to_keep
    }
    for span_output in span_outputs:
        if span_output.output_type == "STRICT":
            for eval_type in ["strict", "ent_type", "partial", "exact"]:
                span_pii_eval[eval_type]["correct"] += 1
                span_ent_eval[span_output.annotated_span.entity_type][
                    eval_type
                ]["correct"] += 1
        elif span_output.output_type == "EXACT":
            for eval_type in ["strict", "ent_type"]:
                span_pii_eval[eval_type]["incorrect"] += 1
                span_ent_eval[span_output.annotated_span.entity_type][
                    eval_type
                ]["incorrect"] += 1
            for eval_type in ["partial", "exact"]:
                span_pii_eval[eval_type]["correct"] += 1
                span_ent_eval[span_output.annotated_span.entity_type][
                    eval_type
                ]["correct"] += 1
        elif span_output.output_type == "ENT_TYPE":
            span_pii_eval["strict"]["incorrect"] += 1
            span_pii_eval["ent_type"]["correct"] += 1
            span_pii_eval["partial"]["partial"] += 1
            span_pii_eval["exact"]["incorrect"] += 1
            span_ent_eval[span_output.annotated_span.entity_type]["strict"][
                "incorrect"
            ] += 1
            span_ent_eval[span_output.annotated_span.entity_type][
                "ent_type"
            ]["correct"] += 1
            span_ent_eval[span_output.annotated_span.entity_type][
                "partial"
            ]["partial"] += 1
            span_ent_eval[span_output.annotated_span.entity_type]["exact"][
                "incorrect"
            ] += 1
        elif span_output.output_type == "PARTIAL":
            for eval_type in ["strict", "ent_type", "exact"]:
                span_pii_eval[eval_type]["incorrect"] += 1
                span_ent_eval[span_output.annotated_span.entity_type][
                    eval_type
                ]["incorrect"] += 1
            span_pii_eval["partial"]["partial"] += 1
            span_ent_eval[span_output.annotated_span.entity_type][
                "partial"
            ]["partial"] += 1
        elif span_output.output_type == "SPURIOUS":
            for eval_type in ["strict", "ent_type", "partial", "exact"]:
                span_pii_eval[eval_type]["spurious"] += 1
                span_ent_eval[span_output.predicted_span.entity_type][
                    eval_type
                ]["spurious"] += 1
        elif span_output.output_type == "MISSED":
            for eval_type in ["strict", "ent_type", "partial", "exact"]:
                span_pii_eval[eval_type]["missed"] += 1
                span_ent_eval[span_output.annotated_span.entity_type][
                    eval_type
                ]["missed"] += 1

    span_ent_eval["PII"] = span_pii_eval
    return span_ent_eval


def span_compute_precision_recall(span_cat_out: Counter, partial_or_type=False) -> \
        Dict[str, int]:
    """
    Takes a result dict that has been output by compute metrics.
    :returns
    dictionary of the results with actual, possible populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """
    span_metrics_out = {"possible": 0, "actual": 0, "precision": 0, "recall": 0}

    correct = span_cat_out["correct"]
    incorrect = span_cat_out["incorrect"]
    partial = span_cat_out["partial"]
    missed = span_cat_out["missed"]
    spurious = span_cat_out["spurious"]

    # Possible: number annotations in the gold-standard
    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system
    actual = correct + incorrect + partial + spurious

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    span_metrics_out["actual"] = actual
    span_metrics_out["possible"] = possible
    span_metrics_out["precision"] = precision
    span_metrics_out["recall"] = recall

    return span_metrics_out


def span_weighted_score(entities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the weighted F2 score where each partial match gets a score based on
    the length of the match
    :param entities_df: pandas DataFrame with span information and their type of match
    :return: pandas DataFrame with weighted F2 score per entity type
    """
    fp = entities_df[entities_df["match_type"] == "SPURIOUS"].shape[0]
    fn = entities_df[entities_df["match_type"] == "MISSED"].shape[0]
    tp_df = entities_df[~entities_df['match_type'].isin(
        ['SPURIOUS', 'MISSED'])].copy()
    weighted_tp = tp_df['overlap_score'].sum()
    precision = weighted_tp / (weighted_tp + fp)
    recall = weighted_tp / (weighted_tp + fn)
    return precision, recall


def span_compute_precision_recall_wrapper(results: dict) -> dict:
    """
    Wraps the compute_precision_recall function and runs on a dict of results
    """

    results_a = {
        key: span_compute_precision_recall(value, True)
        for key, value in results.items()
        if key in ["partial", "ent_type"]
    }
    results_b = {
        key: span_compute_precision_recall(value)
        for key, value in results.items()
        if key in ["strict", "exact"]
    }

    results = {**results_a, **results_b}

    return results


def span_output_to_df(span_outputs: List["SpanOutput"]) -> pd.DataFrame:
    """
    Convert the span output to a pandas DataFrame
    :param span_outputs: List of SpanOutput objects
    :return: pandas DataFrame
    """
    entity_list = []
    for item in span_outputs:
        item_dict = {}
        match_type = item.output_type
        item_dict['match_type'] = match_type
        if match_type == "SPURIOUS":
            item_dict['pred_entity'] = item.predicted_span.entity_type
            item_dict['pred_start'] = item.predicted_span.start_position
            item_dict['pred_end'] = item.predicted_span.end_position
            item_dict['pred_text'] = item.predicted_span.entity_value
            item_dict['overlap_score'] = item.overlap_score
        elif match_type == "MISSED":
            item_dict['true_entity'] = item.annotated_span.entity_type
            item_dict['true_start'] = item.annotated_span.start_position
            item_dict['true_end'] = item.annotated_span.end_position
            item_dict['true_text'] = item.annotated_span.entity_value
            item_dict['overlap_score'] = item.overlap_score
        else:
            item_dict['pred_entity'] = item.predicted_span.entity_type
            item_dict['pred_start'] = item.predicted_span.start_position
            item_dict['pred_end'] = item.predicted_span.end_position
            item_dict['pred_text'] = item.predicted_span.entity_value
            item_dict['true_entity'] = item.annotated_span.entity_type
            item_dict['true_start'] = item.annotated_span.start_position
            item_dict['true_end'] = item.annotated_span.end_position
            item_dict['true_text'] = item.annotated_span.entity_value
            item_dict['overlap_score'] = item.overlap_score
        entity_list.append(item_dict)
    entities_df = pd.DataFrame(entity_list)
    return entities_df


def span_fb_score(precision: float, recall: float, beta: int = 2) -> float:
    """
    Calculate the span F1 score
    :param precision: span precision
    :param recall: span recall
    :param beta: which metric to compute (1 for F1, 2 for F2, etc.)
    """
    try:
        return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    except ZeroDivisionError:
        return None


def span_f1_score(precision: float, recall: float) -> float:
    """
    Calculate the span F1 score
    :param precision: span precision
    :param recall: span recall
    """
    try:
        return 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return None


def align_entity_types(
        input_samples: List[InputSample],
        entities_mapping: Dict[str, str] = None,
        allow_missing_mappings: bool = False,
) -> List[InputSample]:
    """
    Change input samples to conform with Presidio's entities
    :return: new list of InputSample
    """
    new_input_samples = input_samples.copy()

    # A list that will contain updated input samples,
    new_list = []

    for input_sample in new_input_samples:
        contains_field_in_mapping = False
        new_spans = []
        # Update spans to match the entity types in the values of entities_mapping
        for span in input_sample.spans:
            if span.entity_type in entities_mapping.keys():
                new_name = entities_mapping.get(span.entity_type)
                span.entity_type = new_name
                contains_field_in_mapping = True

                new_spans.append(span)
            else:
                if not allow_missing_mappings:
                    raise ValueError(
                        f"Key {span.entity_type} cannot be found "
                        f"in the provided entities_mapping"
                    )
        input_sample.spans = new_spans

        # Update tags in case this sample has relevant entities for evaluation
        if contains_field_in_mapping:
            for i, tag in enumerate(input_sample.tags):
                has_prefix = "-" in tag
                if has_prefix:
                    prefix = tag[:2]
                    clean = tag[2:]
                else:
                    prefix = ""
                    clean = tag

                if clean in entities_mapping.keys():
                    new_name = entities_mapping.get(clean)
                    input_sample.tags[i] = "{}{}".format(prefix, new_name)
                else:
                    input_sample.tags[i] = "O"

        new_list.append(input_sample)
    return new_list
