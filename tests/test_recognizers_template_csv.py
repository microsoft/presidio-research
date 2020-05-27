from presidio_evaluator.data_generator import generate
from presidio_evaluator.presidio_recognizer_evaluator import \
    score_presidio_recognizer
import pytest
import numpy as np

from presidio_analyzer.predefined_recognizers.credit_card_recognizer import CreditCardRecognizer

# test case parameters for tests with dataset generated from a template and csv values
class TemplateTextTestCase:
    def __init__(self, test_name, pii_csv, utterances, dictionary_path,
                 num_of_examples, acceptance_threshold, marks):
        self.test_name = test_name
        self.pii_csv = pii_csv
        self.utterances = utterances
        self.dictionary_path = dictionary_path
        self.num_of_examples = num_of_examples
        self.acceptance_threshold = acceptance_threshold
        self.marks = marks

    def to_pytest_param(self):
        return pytest.param(self.pii_csv, self.utterances, self.dictionary_path,
                            self.num_of_examples, self.acceptance_threshold,
                            id=self.test_name, marks=self.marks)


# template-dataset test cases
cc_test_template_testdata = [
    # large dataset fixture. marked as slow
    TemplateTextTestCase(
        test_name="fake-names-100",
        pii_csv="{}/data/FakeNameGenerator.com_100.csv",
        utterances="{}/data/templates.txt",
        dictionary_path="{}/data/Dictionary_test.csv",
        num_of_examples=100,
        acceptance_threshold=0.9,
        marks=pytest.mark.slow
    )
]


# credit card recognizer tests on template-generates data
@pytest.mark.parametrize("pii_csv, "
                         "utterances, "
                         "dictionary_path, "
                         "num_of_examples, "
                         "acceptance_threshold",
                         [testcase.to_pytest_param()
                          for testcase in cc_test_template_testdata])
def test_credit_card_recognizer_with_template(pii_csv, utterances,
                                              dictionary_path,
                                              num_of_examples,
                                              acceptance_threshold):
    """
        Test credit card recognizer with a dataset generated from
        template and a CSV values file
        :param pii_csv: input csv file location
        :param utterances: template file location
        :param dictionary_path: dictionary/vocabulary file location
        :param num_of_examples: number of samples to be used from dataset
        to test
        :param acceptance_threshold: minimim precision/recall
         allowed for tests to pass
    """

    # read template and CSV files
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))

    input_samples = generate(fake_pii_csv=pii_csv.format(dir_path),
                             utterances_file=utterances.format(dir_path),
                             dictionary_path=dictionary_path.format(dir_path),
                             lower_case_ratio=0.5,
                             num_of_examples=num_of_examples)

    scores = score_presidio_recognizer(
        CreditCardRecognizer(), 'CREDIT_CARD', input_samples)
    if not np.isnan(scores.pii_f):
        assert acceptance_threshold <= scores.pii_f
