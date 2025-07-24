# Span Evaluation Strategy Overview

This document explains how the span evaluation works in Presidio Evaluator, focusing on different overlap scenarios
between annotated and predicted spans.

With span evaluation, there could be multiple different scenarios of span overlaps: Overlap with zero spans, overlap
with one span, overlap with multiple spans. In addition, overlaps could occur with spans of the same entity type, spans
of different types, or one span overlapping with multiple spans from multiple types. This document will break down the
different overlap scenarios, and the expected aggregations on each scenario.

## Key Concepts

- **Span**: A continuous sequence of tokens representing an entity
- **IoU (Intersection over Union)**: Measures the overlap between spans
- **Threshold**: Minimum IoU value to consider a match

## Basic Overlap Scenarios

### 1. Exact Match (Same Type, High IoU)

When a predicted span correctly matches an annotated span:

- **Example**:
    - Text: "John Smith visited Boston"
    - Annotation: [PERSON, PERSON, O, O]
    - Prediction: [PERSON, PERSON, O, O]
- **Result**: True Positive (TP)

### 2. Missing Entity (No Prediction)

When an annotated entity has no corresponding prediction:

- **Example**:
    - Text: "John Smith visited Boston"
    - Annotation: [PERSON, PERSON, O, LOCATION]
    - Prediction: [PERSON, PERSON, O, O]
- **Result**: False Negative (FN) for "Boston"

### 3. False Detection (No Annotation)

When a prediction has no corresponding annotation:

- **Example**:
    - Text: "The report was filed yesterday"
    - Annotation: [O, O, O, O, O]
    - Prediction: [O, ORGANIZATION, O, O, O]
- **Result**: False Positive (FP) for "report"

### 4. Type Mismatch (Different Type, High IoU)

When there's significant overlap but entity types differ:

- **Example**:
    - Text: "New York is a city"
    - Annotation: [LOCATION, LOCATION, O, O, O]
    - Prediction: [ORGANIZATION, ORGANIZATION, O, O, O]
- **Result**: Both FN for LOCATION and FP for ORGANIZATION

### 5. Partial Match (Low IoU)

When there's insufficient overlap between spans:

- **Same Type Example**:
    - Text: "John Smith Johnson visited"
    - Annotation: [PERSON, PERSON, PERSON, O]
    - Prediction: [PERSON, O, O, O]
    - Result: FN (IoU = 0.33, below threshold)
- **Different Type Example**:
    - Text: "New York Mets won"
    - Annotation: [ORGANIZATION, ORGANIZATION, ORGANIZATION, O]
    - Prediction: [LOCATION, O, O, O]
    - Result: FN for ORGANIZATION and FP for LOCATION

## Multiple Span Scenarios

When an annotation overlaps with multiple prediction spans:

### 1. Multiple Spans of Same Type

Spans of the same type are combined, and their collective IoU is calculated:

- **Example**:
    - Text: "New York Mets"
    - Annotation: [ORGANIZATION, ORGANIZATION, ORGANIZATION]
    - Prediction: [ORGANIZATION, O, ORGANIZATION]
    - Combined IoU = 0.67
    - If threshold = 0.5: Treated as a match (TP)
    - If threshold = 0.75: Treated as a miss (FN)

### 2. Multiple Spans of Different Types

Each entity type is evaluated separately against the annotation:

- **Example**:
    - Text: "John Smith Johnson"
    - Annotation: [PERSON, PERSON, PERSON]
    - Prediction: [PERSON, LOCATION, PERSON]
    - PERSON IoU = 0.67, LOCATION IoU = 0.33
    - If threshold = 0.5: PERSON is a match but wrong type for LOCATION portion
    - Result: TP for PERSON, FP for LOCATION

## Real-world Examples

### Example 1: Complex Name with Multiple Parts

- **Text**: "Dr. Jane Smith-Johnson, PhD."
- **Annotation**: [TITLE, PERSON, PERSON, PERSON, O, TITLE]
- **Prediction**: [TITLE, PERSON, PERSON, O, O, O]
- **Result**:
    - TP for TITLE "Dr."
    - TP for PERSON parts "Jane Smith" (combined IoU > threshold)
    - FN for "-Johnson, PhD." parts

### Example 2: Address with Mixed Types

- **Text**: "123 Main Street, New York, NY 10001"
- **Annotation**: [ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS]
- **Prediction**: [ADDRESS, ADDRESS, ADDRESS, LOCATION, LOCATION, LOCATION, ADDRESS]
- **Result**:
    - If the combined IoU for ADDRESS is above threshold, it counts as TP, else FN
    - LOCATION is counted as a FP
