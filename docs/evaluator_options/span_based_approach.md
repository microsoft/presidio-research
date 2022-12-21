# Evaluation at the span level
Besides the evaluation metrics based on a token, this repository provides other scenarios at span level. 

## Evaluation Metrics
To be able to calculate the precision and recall, firstly we compare the golden standard annotations with the output of PII models, then define those different scenarios:


| Senario No | Description | Label |
| ---------- | ----------- | ----- |
| 1 | Entity types and spans match | strict |
| 2 | Entity types match but the spans' boundaries overlapping ratio is between [overlap_threshold, 1) | exact |
| 3 | Entity types are wrong but the spans' boundaries overlapping ratio is between [overlap_threshold, 1] | partial |
| 4 | Regardless of what the predicted entity is if the spans' boundaries overlapping ratio is between (0, overlap_threshold] (*) | incorrect |
| 5 | Regardless of the entity type, the spans' boundaries overlapping ratio is 0 | spurious |
| 6 | The span exist in gold standard annotation but doesn't exist in the predicted outcome | miss |

    (*) Spans' boundaries overlapping ratio = the number of intersecting character between gold and predicted spans / maximum number of characters between gold and predicted spans

    (*) overlap_threshold can be customize in each use case. If it is not providied, our evaluation uses the default value of 0.5

Then, we are able to calculate two additional metrics from those labels:

<b>Possible</b>: The number of annotations in the gold-standard which contributes to the final score:
    Possible = strict + exact + partial + incorrect + missed

<b>Actual</b>: The number of annotations produced by the PII detection system
    Actual = strict + exact + partial + incorrect + spurious

### Metrics calculation for strict matching cases
$$ Precision = \frac{strict}{actual} $$

$$ Recall = \frac{strict}{possible} $$

### Metrics calculation for flexible matching cases
$$ Flexible Precision = \frac{(strict + extract)}{actual} $$

$$ Flexible Recall = \frac{(strict + extract)}{possible} $$

### Metrics calculation for partial matching cases
$$ Partial Precision = \frac{(strict + extract + 0.5 * partial)}{actual} $$

$$ Partial Recall = \frac{(strict + extract + 0.5 * partial)}{possible} $$

An example of the span-level evaluation is summarized in the following diagram:

![span-evaluator](span-evaluator.PNG)