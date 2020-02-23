# Presidio-research
This package features data-science related tasks for developing new recognizers for [Presidio](https://github.com/microsoft/presidio). 
It is used for the evaluation of the entire system, as well as for evaluating specific PII recognizers or PII detection models

## Who should use it?
Anyone interested in evaluating an existing Presidio instance, a specific PII recognizer or to develop new models or logic for detecting PII could leverage the preexisting work in this package.
Additionally, anyone interested in generating new data based on previous datasets (e.g. to increase the coverage of entity values) for Named Entity Recognition models could leverage the data generator contained in this package.

## Getting started
To install the package, clone the repo and install all dependencies, preferably in a virtual environment:

``` sh
# Create conda env (optional)
conda create --name presidio python=3.7
conda activate presidio

# Install package+dependencies
pip install -r requirements.txt
python setup.py install

# Verify installation
pytest
```
Note that some dependencies (such as Flair) are not installed to reduce installation complexity.


## What's in this package?

1. **Data generator** for PII recognizers and NER models
2. **Data representation layer** for data generation, modeling and analysis
3. Multiple **Model/Recognizer evaluation** files (e.g. for Spacy, Flair, CRF, Presidio API, Presidio Analyzer python package, specific Presidio recognizers)
4. **Training and modeling code** for multiple models
4. Helper functions for **results analysis**



## 1. Data generation
See [Data Generator README](/presidio_evaluator/data_generator/README.md) for more details.

The data generation process receives a file with templates, e.g. `My name is [FIRST_NAME]` and a data frame with fake PII data. 
Then, it creates new synthetic sentences by sampling templates and PII values. Furthermore, it tokenizes the data, creates tags (either IO/IOB/BILOU) and spans for the newly created samples.

- For information on data generation/augmentation, see the data generator [README](presidio_evaluator/data_generator/README.md).

- For an example for running the generation process, see [this notebook](notebooks/Generate%20data.ipynb). 

- For an understanding of the underlying fake PII data used, see this [exploratory data analysis notebook](notebooks/PII%20EDA.ipynb).
Note that the generation process might not work off-the-shelf as we are not sharing the fake PII datasets and templates used in this analysis, do to copyright and other restrictions.

Once data is generated, it could be split into train/test/validation sets while ensuring that each template only exists in one set. See [this notebook for more details](notebooks/Split%20by%20pattern%20%23.ipynb).

## 2. Data representation

In order to standardize the process, we use specific data objects that hold all the information needed for generating, analyzing, modeling and evaluating data and models. Specifically, see [data_objects.py](presidio_evaluator/data_objects.py).

## 3. Recognizer evaluation
The presidio-evaluator framework allows you to evaluate Presidio as a system, or a specific PII recognizer for precision and recall.
The main logic lies in the [ModelEvaluator](presidio_evaluator/model_evaluator.py) class. It provides a structured way of evaluating models and recognizers.


### Ready evaluators
Some evaluators were developed for analysis and references. These include:

#### 1. Presidio API evaluator

Allows you to evaluate an existing Presidio deployment through the API. [See this notebook for details](notebooks/Evaluate%20Presidio-API.ipynb).

#### 2. Presidio analyzer evaluator
Allows you to evaluate the local Presidio-Analyzer package. Faster than the API option but requires you to have Presidio-Analyzer installed locally. [See this class for more information](presidio_evaluator/presidio_analyzer.py)

#### 3. One recognizer evaluator
Evaluate one specific recognizer for precision and recall. See [presidio_recognizer_evaluator.py](presidio_evaluator/presidio_recognizer_evaluator.py)


## 4. Modeling

### Conditional Random Fields
To train a CRF on a new dataset, see [this notebook](notebooks/models/CRF.ipynb).
To evaluate a CRF model, see the the [same notebook](notebooks/models/CRF.ipynb) or [this class](presidio_evaluator/crf_evaluator.py).

### spaCy based models
There are three ways of interacting with spaCy models: 
1. Evaluate an existing trained model
2. Train with pretrained embeddings
3. Fine tune an existing spaCy model

Before interacting with spaCy models, the data needs to be adapted to fit spaCy's API. 
See [this notebook for creating spaCy datasets](notebooks/models/Create%20datasets%20for%20Spacy%20training.ipynb).

#### Evaluate an existing trained model
To evaluate spaCy based models, see [this notebook](notebooks/models/Evaluate%20spacy%20models.ipynb).

#### Train with pretrain embeddings
In order to train a new spaCy model from scratch with pretrained embeddings (FastText wiki news subword in this case), follow these three steps:

##### 1. Download FastText pretrained (sub) word embeddings
``` sh
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
unzip wiki-news-300d-1M-subword.vec.zip
```

##### 2. Init spaCy model with pre-trained embeddings
Using spaCy CLI:
``` sh
python -m spacy init-model en spacy_fasttext --vectors-loc wiki-news-300d-1M-subword.vec
```

##### 3. Train spaCy NER model
Using spaCy CLI:
``` sh
python -m spacy train en spacy_fasttext_100 train.json test.json --vectors spacy_fasttext --pipeline ner -n 100
```

#### Fine-tune an existing spaCy model
See [this code for retraining an existing spaCy model](models/spacy_retrain.py). Specifically, run a SpacyRetrainer:
First, you would have to create train and test pickle files for your train and test sets. See [this notebook](notebooks/models/Create%20datasets%20for%20Spacy%20training.ipynb) for more information.

```python
from models import SpacyRetrainer
spacy_retrainer = SpacyRetrainer(original_model_name='en_core_web_lg',
                                 experiment_name='new_spacy_experiment',
                                 n_iter=500, dropout=0.1, aml_config=None)
spacy_retrainer.run()
```

### Flair based models
To train a new model, see the [FlairTrainer](presidio_evaluator/models/flair_train.py) object. 
For experimenting with other embedding types, change the `embeddings` object in the `train` method.
To train a Flair model, run:

```python
from models import FlairTrainer
train_samples = "../data/generated_train.json"
test_samples = "../data/generated_test.json"
val_samples = "../data/generated_validation.json"

trainer = FlairTrainer()
trainer.create_flair_corpus(train_samples, test_samples, val_samples)

corpus = trainer.read_corpus("")
trainer.train(corpus)
```

To evaluate an existing model, see [this notebook](notebooks/models/Evaluate%20flair%20models.ipynb).

# For more information
- [Blog post on NLP approaches to data anonymization](https://towardsdatascience.com/nlp-approaches-to-data-anonymization-1fb5bde6b929)
- [Conference talk about leveraging Presidio and utilizing NLP approaches for data anonymization](https://youtu.be/Tl773LANRwY)

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Copyright notice:

Fake Name Generator identities by the [Fake Name Generator](https://www.fakenamegenerator.com/)
are licensed under a [Creative Commons Attribution-Share Alike 3.0 United States License](http://creativecommons.org/licenses/by-sa/3.0/us/). Fake Name Generator and the Fake Name Generator logo are trademarks of Corban Works, LLC.
