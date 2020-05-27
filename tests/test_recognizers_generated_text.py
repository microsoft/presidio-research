from presidio_evaluator.data_generator import read_synth_dataset
from presidio_evaluator.presidio_recognizer_evaluator import score_presidio_recognizer
import pytest

from presidio_analyzer.predefined_recognizers.credit_card_recognizer import CreditCardRecognizer

# test case parameters for tests with dataset which was previously generated.
class GeneratedTextTestCase:
    def __init__(self, test_name, test_input, acceptance_threshold, marks):
        self.test_name = test_name
        self.test_input = test_input
        self.acceptance_threshold = acceptance_threshold
        self.marks = marks

    def to_pytest_param(self):
        return pytest.param(self.test_input, self.acceptance_threshold,
                            id=self.test_name, marks=self.marks)


# generated-text test cases
cc_test_generate_text_testdata = [
    # small set fixture which expects all type results.
    GeneratedTextTestCase(
        test_name="small-set",
        test_input="{}/data/generated_small.txt",
        acceptance_threshold=1,
        marks=pytest.mark.none
    ),
    # large set fixture which expects all type results. marked as "slow"
    GeneratedTextTestCase(
        test_name="large_set",
        test_input="{}/data/generated_large.txt",
        acceptance_threshold=1,
        marks=pytest.mark.slow
    )
]


# credit card recognizer tests on generated data
@pytest.mark.parametrize("test_input,acceptance_threshold",
                         [testcase.to_pytest_param()
                          for testcase in cc_test_generate_text_testdata])
def test_credit_card_recognizer_with_generated_text(test_input, acceptance_threshold):
    """
        Test credit card recognizer with a generated dataset text file
        :param test_input: input text file location
        :param acceptance_threshold: minimim precision/recall
         allowed for tests to pass
    """

    # read test input from generated file
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = read_synth_dataset(
        test_input.format(dir_path))
    scores = score_presidio_recognizer(
        CreditCardRecognizer(), 'CREDIT_CARD', input_samples)
    assert acceptance_threshold <= scores.pii_f
