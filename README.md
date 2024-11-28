# Presidio-research

This package features data-science related tasks for developing new recognizers for 
[Presidio](https://github.com/microsoft/presidio).
It is used for the evaluation of the entire system, 
as well as for evaluating specific PII recognizers or PII detection models. 
In addition, it contains a fake data generator which creates fake sentences based on templates and fake PII.

## Who should use it?

- Anyone interested in **developing or evaluating PII detection models**, an existing Presidio instance or a Presidio PII recognizer.
- Anyone interested in **generating new data based on previous datasets or sentence templates** (e.g. to increase the coverage of entity values) for Named Entity Recognition models.

## Getting started


### Using notebooks
The easiest way to get started is to look at the notebooks. 
- [Notebook 1](notebooks/1_Generate_data.ipynb): Shows how to use the PII data generator.
- [Notebook 2](notebooks/2_PII_EDA.ipynb): Shows a simple analysis of the PII dataset.
- [Notebook 3](notebooks/3_Split_by_pattern_number.ipynb): Provides tools to split the dataset to train/test/validation while avoiding leakage due to the same pattern appearning in multiple folds (only applicable for synthetically generated data).
- [Notebook 4](notebooks/4_Evaluate_Presidio_Analyzer.ipynb): Shows how to use the evaluation tools to evaluate how well Presidio detects PII. Note that this is using the vanilla Presidio, and the results aren't very accurate.
- [Notebook 5](notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb): Shows how one can configure Presidio to detect PII much more accurately, and boost the f score in ~30%.

### Installation

>Note: Presidio evaluator requires Python>=3.9

#### From PyPI

``` sh
conda create --name presidio python=3.9
conda activate presidio
pip install presidio-evaluator

# Download a spaCy model used by presidio-analyzer
python -m spacy download en_core_web_lg
```

#### From source

To install the package:
1. Clone the repo
2. Install all dependencies, preferably in a virtual environment:

``` sh
# Install package+dependencies
pip install poetry
poetry install --with=dev

# To install with all additional NER dependencies (e.g. Flair, Stanza, CRF), run:
# poetry install --with='ner,dev'

# Download a spaCy model used by presidio-analyzer
python -m spacy download en_core_web_lg

# Verify installation
pytest
```

Note that some dependencies (such as Flair and Stanza) are not automatically installed to reduce installation complexity.

## What's in this package?

1. **Fake data generator** for PII recognizers and NER models
2. **Data representation layer** for data generation, modeling and analysis
3. Multiple **Model/Recognizer evaluation** files (e.g. for Spacy, Flair, CRF, Presidio API, Presidio Analyzer python package, specific Presidio recognizers)
4. **Training and modeling code** for multiple models
5. Helper functions for **results analysis**

## 1. Data generation

See [Data Generator README](presidio_evaluator/data_generator/README.md) for more details.

The data generation process receives a file with templates, e.g. `My name is {{name}}`. 
Then, it creates new synthetic sentences by sampling templates and PII values. 
Furthermore, it tokenizes the data, creates tags (either IO/BIO/BILUO) and spans for the newly created samples.

- For information on data generation/augmentation, see the data generator [README](presidio_evaluator/data_generator/README.md).
- For an example for running the generation process, see [this notebook](notebooks/1_Generate_data.ipynb).
- For an understanding of the underlying fake PII data used, see this [exploratory data analysis notebook](notebooks/2_PII_EDA.ipynb).

Once data is generated, it could be split into train/test/validation sets 
while ensuring that each template only exists in one set. 
See [this notebook for more details](notebooks/3_Split_by_pattern_number.ipynb).

## 2. Data representation

In order to standardize the process, 
we use specific data objects that hold all the information needed for generating, 
analyzing, modeling and evaluating data and models. Specifically, 
see [data_objects.py](presidio_evaluator/data_objects.py).

The standardized structure, `List[InputSample]` could be translated into different formats:
- CONLL
```python
from presidio_evaluator import InputSample
dataset = InputSample.read_dataset_json("data/synth_dataset_v2.json")
conll = InputSample.create_conll_dataset(dataset)
conll.to_csv("dataset.csv", sep="\t")

```

- spaCy v3
```python
from presidio_evaluator import InputSample
dataset = InputSample.read_dataset_json("data/synth_dataset_v2.json")
InputSample.create_spacy_dataset(dataset, output_path="dataset.spacy")
```

- Flair
```python
from presidio_evaluator import InputSample
dataset = InputSample.read_dataset_json("data/synth_dataset_v2.json")
flair = InputSample.create_flair_dataset(dataset)
```

- json
```python
from presidio_evaluator import InputSample
dataset = InputSample.read_dataset_json("data/synth_dataset_v2.json")
InputSample.to_json(dataset, output_file="dataset_json")
```

## 3. PII models evaluation

The presidio-evaluator framework allows you to evaluate Presidio as a system, a NER model, 
or a specific PII recognizer for precision and recall and error-analysis. See [Notebook 5](notebooks/5_Evaluate_Custom_Presidio_Analyzer) for an example.


## 4. Training PII detection models

### CRF

To train a vanilla CRF on a new dataset, see [this notebook](notebooks/models/Train%20CRF.ipynb). To evaluate, see [this notebook](notebooks/models/Evaluate%20CRF%20models.ipynb).

### spaCy

To train a new spaCy model, first save the dataset in a spaCy format:
```python
# dataset is a List[InputSample]
InputSample.create_spacy_dataset(dataset ,output_path="dataset.spacy")
```

To evaluate, see [this notebook](notebooks/models/Evaluate%20spacy%20models.ipynb)

### Flair

- To train Flair models, see this [helper class](presidio_evaluator/models/flair_train.py) or this snippet:
```python
from presidio_evaluator.models import FlairTrainer
train_samples = "data/generated_train.json"
test_samples = "data/generated_test.json"
val_samples = "data/generated_validation.json"

trainer = FlairTrainer()
trainer.create_flair_corpus(train_samples, test_samples, val_samples)

corpus = trainer.read_corpus("")
trainer.train(corpus)
```

> Note that the three json files are created using `InputSample.to_json`.

## For more information


- [Blog post on NLP approaches to data anonymization](https://towardsdatascience.com/nlp-approaches-to-data-anonymization-1fb5bde6b929)
- [Conference talk about leveraging Presidio and utilizing NLP approaches for data anonymization](https://youtu.be/Tl773LANRwY)

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Copyright notice:

Fake Name Generator identities by the [Fake Name Generator](https://www.fakenamegenerator.com/)
are licensed under a [Creative Commons Attribution-Share Alike 3.0 United States License](http://creativecommons.org/licenses/by-sa/3.0/us/).
Fake Name Generator and the Fake Name Generator logo are trademarks of Corban Works, LLC.
