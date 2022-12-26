# Presidio Data Generator

This data generator takes a text file with templates (e.g. `my name is {{person}}`)
and creates a list of InputSamples which contain fake PII entities
instead of placeholders. It further creates spans (start and end of each entity)
for model training and evaluation.

## Scenarios

There are two main scenarios for using the Presidio Data Generator:

1. Create a fake dataset for evaluation or training purposes, given a list of predefined templates 
(see [this file](raw_data/templates.txt) for example)
2. Augment an existing labeled dataset with additional fake values.

In both scenarios the process is similar. In scenario 2, the existing dataset is first translated into templates, 
and then scenario 1 is applied.

## Process

This generator heavily relies on the [Faker package](https://www.github.com/joke2k/faker) with a few differences:

1. `PresidioDataGenerator` returns not only fake text, but also the spans in which fake entities appear in the text.

2. `Faker` samples each value independently. 
In many cases we would want to keep the semantic dependency between two values. 
For example, for the template `My name is {{name}} and my email is {{email}}`, 
we would prefer a result which has the name within the email address, 
such as `My name is Mike and my email is mike1243@gmail.com`. 
For this functionality, a new `RecordGenerator` (based on Faker's `Generator` class) is implemented. 
It accepts a dictionary / pandas DataFrame, and favors returning objects from the same record (if possible).

## Example

For a full example, see the [Generate Data Notebook](../../notebooks/1_Generate_data.ipynb).

Simple example:

```python
from presidio_evaluator.data_generator import PresidioDataGenerator

sentence_templates = [
    "My name is {{name}}",
    "Please send it to {{address}}",
    "I just moved to {{city}} from {{country}}"
]


data_generator = PresidioDataGenerator()
fake_records = data_generator.generate_fake_data(
    templates=sentence_templates, n_samples=10
)

fake_records = list(fake_records)

# Print the spans of the first sample
print(fake_records[0].fake)
print(fake_records[0].spans)



```

The process in high level is the following:

1. Translate a NER dataset (e.g. CONLL or OntoNotes) into a list of
templates: `My name is John` -> `My name is [PERSON]`
2. (Optional) add new Faker providers to the `PresidioDataGenerator` to support types of PII not returned by Faker
3. (Optional) map dataset entity names into provider equivalents by calling `PresidioDataGenerator.add_provider_alias`. 
This will create entity aliases (e.g. faker supports "name" but templates contain "person")
4. Generate samples using the templates list
5. Split the generated dataset to train/test/validation while making sure
that samples from the same template would only appear in one set
6. Adapt datasets for the various models (Spacy, Flair, CRF, sklearn)
7. Train models
8. Evaluate using one of the [evaluation notebooks](../../notebooks/models)

Notes:

- For steps 5, 6, 7 see the main [README](../../README.md).


*Copyright notice:*

Fake Name Generator identities by the Fake Name Generator are licensed under a
Creative Commons Attribution-Share Alike 3.0 United States License.
Fake Name Generator and the Fake Name Generator logo
are trademarks of Corban Works, LLC.
