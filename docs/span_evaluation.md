# Span Evaluation in Presidio Evaluator

This document explains the span evaluation process implemented in Presidio Evaluator, covering how spans are created,
matched, and evaluated, along with comparisons to other evaluation paradigms.

## Span Creation and Processing

### Span Creation

Spans are created from token-level annotations in the input data. Each span represents a continuous sequence of tokens
with the same entity type annotation. The basic properties of a span include:

- `entity_type`: The type of entity (e.g., PERSON, LOCATION)
- `entity_value`: The actual text of the entity
- `start_position` and `end_position`: Character-level boundaries


#### Span Normalization

For more advanced processing, spans also include normalized versions of the text:

- `normalized_tokens`: List of normalized tokens that make up the span (typically lowercased and with special characters
  removed)
- `normalized_start_index` and `normalized_end_index`: Character indices in the normalized text

Normalization helps with more consistent matching between variations of the same entity (e.g., "John Smith" vs "john
smith") and by better handling of punctuation marks and skip words.

### Span Merging

In some cases, multiple separate tokens may need to be merged into a single span:

1. **Adjacent tokens of same type**: Consecutive tokens with the same entity type are merged into a single span
2. **Skip words handling**: Certain configurable words (like punctuation marks or skip words) can be included in spans even if
   they are annotated as non-entities, allowing for more natural entity boundaries

Example of skip words:

```
Text: "University of Washington"
Without skip words: [ORG, O, ORG]
With "of" as skip word: [ORG, ORG, ORG] (treated as one span)
```

The `skip_words` parameter in the `SpanEvaluator` constructor determines which words can be skipped when merging
adjacent spans of the same entity type.

## Span Matching Strategy

The evaluator compares annotation spans (gold standard) with prediction spans (model output) using an Intersection over
Union (IoU) approach. This can be either character-based or token-based, controlled by the `char_based` parameter.

### IoU Calculation

- **Character-based IoU**: Calculates the character overlap between spans
- **Token-based IoU**: Calculates the token overlap between spans

IoU = (Intersection) / (Union)

An IoU threshold (default: 0.75) determines whether spans match sufficiently.

### Matching Annotations to Predictions

The matching process follows these steps:

1. For each annotation span, find all overlapping prediction spans
2. Group overlapping prediction spans by entity type
3. Calculate combined IoU for each entity type group
4. Determine match status based on IoU threshold and entity type

> For a detailed breakdown of different matching scenarios and examples, see
> the [Span Matching Strategies](span_matching_strategies.md) document.

## Metric Calculation

The evaluator calculates both per-entity-type metrics and global PII metrics:

### Per-Entity-Type Metrics

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F-beta**: (1 + beta²) * (precision * recall) / (beta² * precision + recall)

### Global PII Metrics

- **PII Precision**: Total TP / Total predicted spans
- **PII Recall**: Total TP / Total annotated spans
- **PII F-score**: Combined F-score for all entity types

## Comparison with Other Evaluation Paradigms

### Token-level Evaluation vs. Span Evaluation

Token-level evaluation:

- Evaluates each token individually
- Can't capture entity boundaries correctly
- May artificially inflate metrics for multi-token entities

Span evaluation:

- Evaluates complete named entities
- Captures entity boundaries
- More accurately reflects real-world information extraction performance

