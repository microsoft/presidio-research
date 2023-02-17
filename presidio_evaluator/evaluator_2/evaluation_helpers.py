from collections import Counter


def get_actual_possible_span(results: Counter) -> Counter:
    """
    Takes a result dict that has been output by compute metrics.
    :returns
    dictionary of the results with actual, possible populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """
    correct = results["correct"]
    incorrect = results["incorrect"]
    partial = results["partial"]
    missed = results["missed"]
    spurious = results["spurious"]

    # Possible: number annotations in the gold-standard
    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system
    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results


def span_compute_precision_recall(results: dict, partial_or_type=False) -> dict:
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results["partial"]
    correct = results["correct"]

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall

    return results


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
