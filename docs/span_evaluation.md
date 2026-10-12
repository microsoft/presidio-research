# Span Evaluation in Presidio Evaluator

This document explains the span evaluation process implemented in Presidio Evaluator, covering how spans are created,
matched, and evaluated, along with comparisons to other evaluation paradigms.

## Span Creation and Processing

### Span Creation

A span is one entity in the text, such as a name, a date or an email address, and
may cover several tokens. Each sentence has two independent sets of spans: the
annotated spans (the ground truth) and the predicted spans (what the model found).
The two sets can differ in number, position and type, and may not overlap at all.

Presidio Evaluator scores spans at three levels of detail:

- **Binary**: is it PII or not.
- **Branch**: a canonical entity such as person or location.
- **Detailed**: a more granular entity type, if one exists in the dataset or is
  produced by the model.

Where each span starts and ends is decided once, from the original data before
entity types are converted to any of these levels, and kept at every level. The
results data records it on every token, in the `annotation_span_id` and
`prediction_span_id` columns, so the evaluator knows which tokens belong to the
same entity.

Each span has:

- `entity_type`: The type of entity (e.g., PERSON, LOCATION)
- `entity_value`: The actual text of the entity
- `start_position` and `end_position`: Character-level boundaries

!!! note
    In most cases the dataset and the model mark where each entity begins and ends.
    When they don't, two adjacent entities of the same type are read as one span.
    For example, in "Invite Alice Bob", if both names are labeled `PERSON` and
    nothing marks where one ends, they are evaluated as a single span, "Alice Bob".

#### Span Normalization

For more advanced processing, spans also include normalized versions of the text:

- `normalized_tokens`: List of normalized tokens that make up the span (typically lowercased and with special characters
  removed)
- `normalized_start_index` and `normalized_end_index`: Character indices in the normalized text

Normalization helps with more consistent matching between variations of the same entity (e.g., "John Smith" vs "john
smith") and by better handling of punctuation marks and skip words.

#### Skip Words

Skip words (punctuation, stop words) are ignored when comparing spans. A span made
only of skip words is dropped. Skip words do not join separate spans: in
`"Michael and Bob"`, with `and` as a skip word, `Michael` and `Bob` stay two spans.

## Span Matching Strategy

The evaluator compares annotation spans (gold standard) with prediction spans (model output) using an Intersection over
Union (IoU) approach. This can be either character-based or token-based, controlled by the `char_based` parameter.

### IoU Calculation

- **Character-based IoU**: Calculates the character overlap between spans
- **Token-based IoU**: Calculates the token overlap between spans

IoU = (Intersection) / (Union)

An IoU threshold determines whether spans match sufficiently.

### Matching Annotations to Predictions

The matching process follows these steps:

1. Assign each prediction to at most one overlapping annotation: prefer a same-type
   match meeting the IoU threshold, then the greatest IoU. Ties prefer the same
   type, then the earliest gold start and end, then the entity type in descending
   lexical order. This assignment is independent of annotation input order.
2. Group overlapping prediction spans by entity type
3. Calculate combined IoU for each entity type group
4. Determine match status based on IoU threshold and entity type

> For a detailed breakdown of different matching scenarios and examples, see
> the [Span Matching Strategies](span_matching_strategies.md) document.

## Metric Calculation

The evaluator calculates both per-entity-type metrics and global PII metrics:

### Per-Entity-Type Metrics

- **Precision**: TP / num_predicted
- **Recall**: TP / num_annotated
- **F-beta**: (1 + beta²) * (precision * recall) / (beta² * precision + recall)

### Global PII Metrics

- Treat every entity type as if it were a single PII type
- Calculate global precision, recall, and F-score on PII/not PII values


## Evaluation Process

1. Assign predictions to annotations using the exclusive assignment described above
2. Group overlapping spans by entity type
3. Calculate combined IoU for each group
4. Determine match status based on IoU and entity type
5. Mark remaining predictions (with no overlap) as FPs

See more info on the [Span Matching Strategies](span_matching_strategies.md) document.

## Counting Strategy

- Multiple predictions of the same type overlapping with one annotation count as a single prediction
- A prediction participates in only one annotation's group. It cannot be counted
  as a TP or FP again for another gold span. This preserves fragment aggregation;
  `num_predicted` is still a count of evaluated groups, not always raw model spans.
- Different entity types are counted separately
- An annotation is only counted once as annotated (denominator for precision and recall),
  regardless of how many types intersect with it

