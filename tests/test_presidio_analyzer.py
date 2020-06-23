import pytest

from presidio_evaluator import InputSample, Span
from presidio_evaluator.data_generator import read_synth_dataset
from presidio_evaluator.presidio_analyzer_evaluator import PresidioAnalyzerEvaluator

# Mapping between dataset entities and Presidio entities. Key: Dataset entity, Value: Presidio entity
entities_mapping = {
    "PERSON": "PERSON",
    "EMAIL": "EMAIL_ADDRESS",
    "CREDIT_CARD": "CREDIT_CARD",
    "FIRST_NAME": "PERSON",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "BIRTHDAY": "DATE_TIME",
    "DATE": "DATE_TIME",
    "DOMAIN": "DOMAIN",
    "CITY": "LOCATION",
    "ADDRESS": "LOCATION",
    "IBAN": "IBAN_CODE",
    "URL": "DOMAIN_NAME",
    "US_SSN": "US_SSN",
    "IP_ADDRESS": "IP_ADDRESS",
    "ORGANIZATION": "ORG",
    "O": "O",
}


class GeneratedTextTestCase:
    def __init__(self, test_name, test_input, acceptance_threshold, marks):
        self.test_name = test_name
        self.test_input = test_input
        self.acceptance_threshold = acceptance_threshold
        self.marks = marks

    def to_pytest_param(self):
        return pytest.param(
            self.test_input,
            self.acceptance_threshold,
            id=self.test_name,
            marks=self.marks,
        )


# generated-text test cases
analyzer_test_generate_text_testdata = [
    # small set fixture which expects all results.
    GeneratedTextTestCase(
        test_name="small-set",
        test_input="{}/data/generated_small.txt",
        acceptance_threshold=0.3,
        marks=pytest.mark.none,
    )
]


def test_analyzer_simple_input():
    model = PresidioAnalyzerEvaluator(entities_to_keep=["PERSON"])

    sample = InputSample(
        full_text="My name is Mike",
        masked="My name is [PERSON]",
        spans=[Span("PERSON", "Mike", 10, 14)],
        create_tags_from_span=True,
    )

    evaluated = model.evaluate_sample(sample)
    metrics = model.calculate_score([evaluated])

    assert metrics.pii_precision == 1
    assert metrics.pii_recall == 1


# analyzer tests on generated data
@pytest.mark.parametrize(
    "test_input,acceptance_threshold",
    [testcase.to_pytest_param() for testcase in analyzer_test_generate_text_testdata],
)
def test_analyzer_with_generated_text(test_input, acceptance_threshold):
    """
        Test analyzer with a generated dataset text file
        :param test_input: input text file location
        :param acceptance_threshold: minimim precision/recall
         allowed for tests to pass
    """
    # read test input from generated file

    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = read_synth_dataset(test_input.format(dir_path))

    updated_samples = PresidioAnalyzerEvaluator.align_input_samples_to_presidio_analyzer(
        input_samples=input_samples, entities_mapping=entities_mapping
    )

    analyzer = PresidioAnalyzerEvaluator()
    evaluated_samples = analyzer.evaluate_all(updated_samples)
    scores = analyzer.calculate_score(evaluation_results=evaluated_samples)

    assert acceptance_threshold <= scores.pii_precision
    assert acceptance_threshold <= scores.pii_recall
