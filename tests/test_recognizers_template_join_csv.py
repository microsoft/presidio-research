from presidio_evaluator.data_generator import FakeDataGenerator
from presidio_evaluator.presidio_recognizer_evaluator import \
    score_presidio_recognizer
import pandas as pd
import pytest
import numpy as np

from presidio_analyzer import Pattern, PatternRecognizer

# test case parameters for tests with dataset generated from a template and
# two csv value files, one containing the common-entities and another one with custom entities
class PatternRecognizerTestCase:
    def __init__(self, test_name, entity_name, pattern, score, pii_csv, ext_csv,
                 utterances, dictionary_path, num_of_examples, acceptance_threshold,
                 max_mistakes_number, marks):
        self.test_name = test_name
        self.entity_name = entity_name
        self.pattern = pattern
        self.score = score
        self.pii_csv = pii_csv
        self.ext_csv = ext_csv
        self.utterances = utterances
        self.dictionary_path = dictionary_path
        self.num_of_examples = num_of_examples
        self.acceptance_threshold = acceptance_threshold
        self.max_mistakes_number = max_mistakes_number
        self.marks = marks

    def to_pytest_param(self):
        return pytest.param(self.pii_csv, self.ext_csv, self.utterances,
                            self.dictionary_path,
                            self.entity_name, self.pattern, self.score,
                            self.num_of_examples, self.acceptance_threshold,
                            self.max_mistakes_number, id=self.test_name,
                            marks=self.marks)


# template-dataset test cases
rocket_test_template_testdata = [
    # large dataset fixture. marked as slow.
    # all input is correct, test is conclusive
    PatternRecognizerTestCase(
        test_name="rocket-no-errors",
        entity_name="ROCKET",
        pattern=r'\W*(rocket)\W*',
        score=0.8,
        pii_csv="{}/data/FakeNameGenerator.com_100.csv",
        ext_csv="{}/data/FakeRocketGenerator.csv",
        utterances="{}/data/rocket_example_sentences.txt",
        dictionary_path="{}/data/Dictionary_test.csv",
        num_of_examples=100,
        acceptance_threshold=1,
        max_mistakes_number=0,
        marks=pytest.mark.slow
    ),
    # large dataset fixture. marked as slow
    # all input is correct, test is conclusive
    PatternRecognizerTestCase(
        test_name="rocket-all-errors",
        entity_name="ROCKET",
        pattern=r'\W*(rocket)\W*',
        score=0.8,
        pii_csv="{}/data/FakeNameGenerator.com_100.csv",
        ext_csv="{}/data/FakeRocketErrorsGenerator.csv",
        utterances="{}/data/rocket_example_sentences.txt",
        dictionary_path="{}/data/Dictionary_test.csv",
        num_of_examples=100,
        acceptance_threshold=0,
        max_mistakes_number=100,
        marks=pytest.mark.slow
    ),
    # large dataset fixture. marked as slow
    # some input is correct some is not, test is inconclusive
    PatternRecognizerTestCase(
        test_name="rocket-some-errors",
        entity_name="ROCKET",
        pattern=r'\W*(rocket)\W*',
        score=0.8,
        pii_csv="{}/data/FakeNameGenerator.com_100.csv",
        ext_csv="{}/data/FakeRocket50PercentErrorsGenerator.csv",
        utterances="{}/data/rocket_example_sentences.txt",
        dictionary_path="{}/data/Dictionary_test.csv",
        num_of_examples=100,
        acceptance_threshold=0.3,
        max_mistakes_number=70,
        marks=[pytest.mark.slow, pytest.mark.inconclusive]
    )
]


@pytest.mark.parametrize(
    "pii_csv, ext_csv, utterances, dictionary_path, "
    "entity_name, pattern, score, num_of_examples, "
    "acceptance_threshold, max_mistakes_number",
    [testcase.to_pytest_param()
     for testcase in rocket_test_template_testdata])
def test_pattern_recognizer(pii_csv, ext_csv, utterances, dictionary_path,
                            entity_name, pattern,
                            score, num_of_examples, acceptance_threshold,
                            max_mistakes_number):
    """
        Test generic pattern recognizer with a dataset generated from template, a CSV values file with common entities
        and another CSV values file with a custom entity
        :param pii_csv: input csv file location with the common entities
        :param ext_csv: input csv file location with custom entities
        :param utterances: template file location
        :param dictionary_path: vocabulary/dictionary file location
        :param entity_name: custom entity name
        :param pattern: recognizer pattern
        :param num_of_examples: number of samples to be used from dataset to test
        :param acceptance_threshold: minimim precision/recall
         allowed for tests to pass
    """

    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dfpii = pd.read_csv(pii_csv.format(dir_path), encoding='utf-8')
    dfext = pd.read_csv(ext_csv.format(dir_path), encoding='utf-8')
    dictionary_path = dictionary_path.format(dir_path)
    ext_column_name = dfext.columns[0]

    def get_from_ext(i):
        index = i % dfext.shape[0]
        return dfext.iat[index, 0]

    # extend pii with ext data
    dfpii[ext_column_name] = [get_from_ext(i) for i in range(0, dfpii.shape[0])]

    # generate examples
    generator = FakeDataGenerator(fake_pii_csv_file=dfpii,
                                  utterances_file=utterances.format(dir_path),
                                  dictionary_path=dictionary_path)
    examples = generator.sample_examples(num_of_examples)

    pattern = Pattern("test pattern", pattern, score)
    pattern_recognizer = PatternRecognizer(entity_name,
                                           name="test recognizer",
                                           patterns=[pattern])

    scores = score_presidio_recognizer(
        pattern_recognizer, [entity_name], examples)
    if not np.isnan(scores.pii_f):
        assert acceptance_threshold <= scores.pii_f
    assert max_mistakes_number >= len(scores.model_errors)
