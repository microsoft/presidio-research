import datetime
import json

import pandas as pd

from presidio_evaluator import InputSample
from presidio_evaluator.data_generator import FakeDataGenerator


def read_utterances(utterances_file):
    with open(utterances_file) as f:
        return f.readlines()


def generate(fake_pii_csv,
             utterances_file,
             output_file=None,
             num_of_examples=1000,
             dictionary_path=None,
             store_masked_text=False,
             keep_only_tagged=False,
             **kwargs):
    """

    :param fake_pii_csv: csv containing fake PII
    :param utterances_file: txt file containing template sentences
    :param output_file: filepath for json or csv output
    :param num_of_examples: number of examples to generate
    :param dictionary_path: path to vocabulary file
    :param store_masked_text: Whether to remove or keep masked version of text
    :param keep_only_tagged: Ignore utterances with no entity
    (e.g. Remove: 'I went to the shop today', Keep: '[PERSON] went to the shop today')
    :return: list of generated InputSamples
    """

    if not output_file:
        raise ValueError("Please provide an output file path")

    templates = read_utterances(utterances_file)

    if keep_only_tagged:
        templates = [template for template in templates if "[" in template]

    df = pd.read_csv(fake_pii_csv, encoding='utf-8')

    generator = FakeDataGenerator(fake_pii_df=df,
                                  dictionary_path=dictionary_path,
                                  templates=templates, **kwargs)
    counter = 0

    examples = []
    for example in generator.sample_examples(num_of_examples):
        if not store_masked_text:
            example.masked = None
        examples.append(example)

    examples_json = [example.to_dict() for example in examples]

    with open("{}".format(output_file), 'w+', encoding='utf-8') as f:
        json.dump(examples_json, f, ensure_ascii=False, indent=4)

    print("generated {} examples".format(len(examples)))
    print("Finished creating generated dataset. File location:{}".format(output_file))

    return examples


def read_synth_dataset(filepath=None, length=None):
    import json
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if length:
        dataset = dataset[:length]

    input_samples = [InputSample.from_json(row) for row in dataset]

    return input_samples


if __name__ == "__main__":

    # PARAMS:
    EXAMPLES = 30
    PII_FILE_SIZE = 3000
    SPAN_TO_TAG = True
    TEMPLATES_FILE = 'raw_data/templates.txt'
    KEEP_ONLY_TAGGED = False
    LOWER_CASE_RATIO = 0.1
    IGNORE_TYPES = {"IP_ADDRESS", 'US_SSN', 'URL'}

    cur_time = datetime.date.today().strftime("%B %d %Y")
    OUTPUT = "generated_size_{}_date_{}.txt".format(EXAMPLES, cur_time)

    fake_pii_csv = '../../presidio_evaluator/data_generator/' \
                   'raw_data/FakeNameGenerator.com_{}.csv'.format(PII_FILE_SIZE)
    utterances_file = TEMPLATES_FILE
    dictionary_path = None

    examples = generate(fake_pii_csv=fake_pii_csv,
                        utterances_file=utterances_file,
                        dictionary_path=dictionary_path,
                        output_file=OUTPUT,
                        lower_case_ratio=LOWER_CASE_RATIO,
                        num_of_examples=EXAMPLES,
                        ignore_types=IGNORE_TYPES,
                        keep_only_tagged=KEEP_ONLY_TAGGED,
                        span_to_tag=SPAN_TO_TAG)

    # sanity
    input_samples = read_synth_dataset(OUTPUT)
    for sample in input_samples:
        if len(sample.tags) != len(sample.tokens):
            print("ERROR during generation. sample: {}".format(sample))

    print(input_samples[:10])
