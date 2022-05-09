---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: presidio
    language: python
    name: presidio
---

Evaluate ...XYZ... models for person names, orgs and locations using the Presidio Evaluator framework

```python
from pathlib import Path
from copy import deepcopy
from pprint import pprint
from collections import Counter

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import Evaluator, ModelError
from presidio_evaluator.experiment_tracking import get_experiment_tracker
from presidio_evaluator.models import ...Model...

import pandas as pd

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_colwidth', None)

%reload_ext autoreload
%autoreload 2
```

Select data for evaluation:

```python
dataset_name = "synth_dataset_v2.json"
dataset = InputSample.read_dataset_json(Path(Path.cwd().parent.parent, "data", dataset_name))
print(len(dataset))
```

```python
entity_counter = Counter()
for sample in dataset:
    for tag in sample.tags:
        entity_counter[tag] += 1
```

```python
print("Count per entity:")
pprint(entity_counter.most_common())

print("\nExample sentence:")
print(dataset[1])

print("\nMin and max number of tokens in dataset:")
print(f"Min: {min([len(sample.tokens) for sample in dataset])}, " \
      f"Max: {max([len(sample.tokens) for sample in dataset])}")

print("\nMin and max sentence length in dataset:")
print(f"Min: {min([len(sample.full_text) for sample in dataset])}, " \
      f"Max: {max([len(sample.full_text) for sample in dataset])}")
```

Select models for evaluation:

```python
models = [...MODEL NAMES...]
```

Run evaluation on all models:

```python
for model_name in models:
    print("-----------------------------------")
    print(f"Evaluating model {model_name}")
    experiment = get_experiment_tracker()
    
    model = Model(..., entities_to_keep=['PERSON', 'GPE', 'ORG', 'NORP'])
    evaluator = Evaluator(model=model)
    evaluation_results = evaluator.evaluate_all(deepcopy(dataset))
    results = evaluator.calculate_score(evaluation_results)
    
    # update params tracking
    params = {"dataset_name":dataset_name, "model_name": model_name}
    params.update(model.to_log())
    experiment.log_parameters(params)
    experiment.log_dataset_hash(dataset)
    experiment.log_metrics(results.to_log())
    entities, confmatrix = results.to_confusion_matrix()
    experiment.log_confusion_matrix(matrix=confmatrix, labels=entities)
    
    print("Confusion matrix:")
    print(pd.DataFrame(confmatrix, columns=entities, index=entities))
    
    print("Precision and recall")
    print(results)
    
    # end experiment
    experiment.end()
```

### Results analysis

```python
sent = 'I am taiwanese but I live in Cambodia.'
#sent = input("Enter sentence: ")
model.predict(InputSample(full_text=sent))
```

### Error Analysis

```python
errors = results.model_errors
```

#### False positives


1. Most false positive tokens:

```python
ModelError.most_common_fp_tokens(errors)
```

```python
fps_df = ModelError.get_fps_dataframe(errors, entity=["GPE"])
fps_df[["full_text", "token", "prediction"]]
```

2. False negative examples

```python
errors = scores.model_errors
ModelError.most_common_fn_tokens(errors, n=50, entity=["PERSON"])
```

More FN analysis

```python
fns_df = ModelError.get_fns_dataframe(errors, entity=['GPE'])
```

```python
fns_df[["full_text", "token", "annotation", "prediction"]]
```

```python
print("All errors:\n")
[print(error,"\n") for error in errors]
```

```python

```
