# PII dataset generator
This data generator takes a text file with templates (e.g. `my name is [PERSON]`) and creates a list of InputSamples which contain fake PII entities instead of placeholders.
It also creates Spans (start and end of each entity), tokens (`spaCy` tokenizer) and tags in various schemas (BIO/IOB, IO, BILOU)
In addition it provides some off-the-shelf features on each token, like `pos`, `dep` and `is_in_vocabulary`

The main class is `FakeDataGenerator` however the `main` module has two functions for creating and reading a fake dataset.
During the generation process, the tool either takes fake PII from a provided CSV with a known format, and/or from extension functions which can be found in the extensions.py file.

The process in high level is the following:
1. Translate a NER dataset (e.g. CONLL or OntoNotes) into a list of templates: `My name is John` -> `My name is [PERSON]`
2. (Optional) adapt the FakeDataGenerator to support new extensions which could generate fake PII entities
3. Generate X samples using the templates list + a fake PII dataset + extensions that add additional PII entities
4. Split the generated dataset to train/test/validation while making sure that samples from the same template would only appear in one set
5. Adapt datasets for the various models (Spacy, Flair, CRF, sklearn)
6. Train models
7. Evaluate using the evaluation notebooks and using the Presidio Evaluator framework



Notes:
- For steps 5, 6, 7 see the main [README](../../README.md).
- For a simple data generation pipeline, [see this notebook](../../notebooks/Generate data.ipynb).
- For information on transforming a NER dataset into a templates, see the notebooks in the [helper notebooks](helper%20notebooks) folder.

Example run:

```python
TEMPLATES_FILE = 'raw_data/templates.txt'
OUTPUT = "generated_.txt"

## Should be downloaded from FakeNameGenerator
fake_pii_csv = 'raw_data/FakeNameGenerator.csv'

examples = generate(fake_pii_csv=fake_pii_csv,
                    utterances_file=TEMPLATES_FILE,
                    dictionary_path=None,
                    output_file=OUTPUT,
                    lower_case_ratio=0.1,
                    num_of_examples=100,
                    ignore_types={"IP_ADDRESS", 'US_SSN', 'URL'},
                    keep_only_tagged=False,
                    span_to_tag=True)
```


*Copyright notice:*

Fake Name Generator identities by the Fake Name Generator are licensed under a Creative Commons Attribution-Share Alike 3.0 United States License. Fake Name Generator and the Fake Name Generator logo are trademarks of Corban Works, LLC.