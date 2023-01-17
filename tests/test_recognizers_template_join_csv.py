from presidio_evaluator import InputSample
from presidio_evaluator.data_generator import PresidioSentenceFaker
from presidio_evaluator.evaluation.scorers import score_presidio_recognizer
import pandas as pd
import pytest
import numpy as np

from presidio_analyzer import Pattern, PatternRecognizer


class PatternRecognizerTestCase:
    """
    Test case parameters for tests with dataset generated from a template and
    two csv value files, one containing the common-entities and another one with custom entities.
    """

    def __init__(
        self,
        test_name,
        entity_name,
        pattern,
        score,
        pii_csv,
        ext_csv,
        utterances,
        num_of_examples,
        acceptance_threshold,
        max_mistakes_number,
    ):
        self.test_name = test_name
        self.entity_name = entity_name
        self.pattern = pattern
        self.score = score
        self.pii_csv = pii_csv
        self.ext_csv = ext_csv
        self.utterances = utterances
        self.num_of_examples = num_of_examples
        self.acceptance_threshold = acceptance_threshold
        self.max_mistakes_number = max_mistakes_number

    def to_pytest_param(self):
        return pytest.param(
            self.pii_csv,
            self.ext_csv,
            self.utterances,
            self.entity_name,
            self.pattern,
            self.score,
            self.num_of_examples,
            self.acceptance_threshold,
            self.max_mistakes_number,
            id=self.test_name,
        )


# template-dataset test cases
rocket_test_template_testdata = [
    PatternRecognizerTestCase(
        test_name="rocket-no-errors",
        entity_name="ROCKET",
        pattern=r"\W*(rocket)\W*",
        score=0.8,
        pii_csv="{}/data/FakeNameGenerator.com_100.csv",
        ext_csv="{}/data/FakeRocketGenerator.csv",
        utterances="{}/data/rocket_example_sentences.txt",
        num_of_examples=100,
        acceptance_threshold=1,
        max_mistakes_number=0,
    ),
    PatternRecognizerTestCase(
        test_name="rocket-all-errors",
        entity_name="ROCKET",
        pattern=r"\W*(rocket)\W*",
        score=0.8,
        pii_csv="{}/data/FakeNameGenerator.com_100.csv",
        ext_csv="{}/data/FakeRocketErrorsGenerator.csv",
        utterances="{}/data/rocket_example_sentences.txt",
        num_of_examples=100,
        acceptance_threshold=0,
        max_mistakes_number=100,
    ),
    PatternRecognizerTestCase(
        test_name="rocket-some-errors",
        entity_name="ROCKET",
        pattern=r"\W*(rocket)\W*",
        score=0.8,
        pii_csv="{}/data/FakeNameGenerator.com_100.csv",
        ext_csv="{}/data/FakeRocket50PercentErrorsGenerator.csv",
        utterances="{}/data/rocket_example_sentences.txt",
        num_of_examples=100,
        acceptance_threshold=0.3,
        max_mistakes_number=70,
    ),
]


@pytest.mark.parametrize(
    "pii_csv, ext_csv, utterances, "
    "entity_name, pattern, score, num_of_examples, "
    "acceptance_threshold, max_mistakes_number",
    [testcase.to_pytest_param() for testcase in rocket_test_template_testdata],
)
def test_pattern_recognizer(
    pii_csv,
    ext_csv,
    utterances,
    entity_name,
    pattern,
    score,
    num_of_examples,
    acceptance_threshold,
    max_mistakes_number,
):
    """
    Test generic pattern recognizer with a dataset generated from template, a CSV values file with common entities
    and another CSV values file with a custom entity
    :param pii_csv: input csv file location with the common entities
    :param ext_csv: input csv file location with custom entities
    :param utterances: template file location
    :param entity_name: custom entity name
    :param pattern: recognizer pattern
    :param num_of_examples: number of samples to be used from dataset to test
    :param acceptance_threshold: minimum precision/recall
     allowed for tests to pass
    """

    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dfpii = pd.read_csv(pii_csv.format(dir_path), encoding="utf-8")
    dfext = pd.read_csv(ext_csv.format(dir_path), encoding="utf-8")
    ext_column_name = dfext.columns[0]

    def get_from_ext(i):
        index = i % dfext.shape[0]
        return dfext.iat[index, 0]

    # extend pii with ext data
    dfpii[ext_column_name] = [get_from_ext(i) for i in range(0, dfpii.shape[0])]

    templates = utterances.format(dir_path)
    sentence_faker = PresidioSentenceFaker('en_US', lower_case_ratio=0.05, sentence_templates=templates)
    examples = sentence_faker.generate_new_fake_sentences(num_of_examples)
    input_samples = [
        InputSample.from_faker_spans_result(example) for example in examples
    ]

    pattern = Pattern("test pattern", pattern, score)
    pattern_recognizer = PatternRecognizer(
        entity_name, name="test recognizer", patterns=[pattern]
    )

    scores = score_presidio_recognizer(
        recognizer=pattern_recognizer,
        entities_to_keep=[entity_name],
        input_samples=input_samples,
    )
    if not np.isnan(scores.pii_f):
        assert acceptance_threshold <= scores.pii_f
    assert max_mistakes_number >= len(scores.model_errors)
