from collections import Counter
import pytest

import presidio_evaluator.evaluator_2.evaluation_helpers as evaluation_helpers


@pytest.mark.parametrize(
    "span_result_input, span_result_output",
    [
        (Counter({'incorrect': 3, 'correct': 2, 'missed': 1, 'spurious': 1, 'partial': 0}),
         Counter({'incorrect': 3, 'correct': 2, 'missed': 1, 'spurious': 1, 'partial': 0, 'possible': 6, 'actual': 6})),
        (Counter({'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 0}),
         Counter({'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 0, 'actual': 0, 'possible': 1})),
        (Counter({'correct': 1, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 0}),
         Counter({'correct': 1, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 0, 'actual': 1, 'possible': 2}))
    ])
def test_get_actual_possible_span_case_1(span_result_input, span_result_output):
    actual_output = evaluation_helpers.get_actual_possible_span(span_result_input)
    assert actual_output == span_result_output


def test_get_actual_possible_span_case2():
    span_pii_eval = {'strict': Counter({'incorrect': 3, 'correct': 2, 'missed': 1, 'spurious': 1, 'partial': 0}),
                     'ent_type': Counter({'correct': 3, 'incorrect': 2, 'missed': 1, 'spurious': 1, 'partial': 0}),
                     'partial': Counter({'correct': 3, 'partial': 2, 'missed': 1, 'spurious': 1, 'incorrect': 0}),
                     'exact': Counter({'correct': 3, 'incorrect': 2, 'missed': 1, 'spurious': 1, 'partial': 0})}

    expected_span_pii_eval = {
        'strict': Counter({'actual': 6, 'possible': 6, 'incorrect': 3, 'correct': 2, 'missed': 1, 'spurious': 1, 'partial': 0}),
        'ent_type': Counter({'actual': 6, 'possible': 6, 'correct': 3, 'incorrect': 2, 'missed': 1, 'spurious': 1, 'partial': 0}),
        'partial': Counter({'actual': 6, 'possible': 6, 'correct': 3, 'partial': 2, 'missed': 1, 'spurious': 1, 'incorrect': 0}),
        'exact': Counter({'actual': 6, 'possible': 6, 'correct': 3, 'incorrect': 2, 'missed': 1, 'spurious': 1, 'partial': 0})}

    for eval_type in span_pii_eval:
        span_pii_eval[eval_type] = evaluation_helpers.get_actual_possible_span(span_pii_eval[eval_type])

    assert span_pii_eval == expected_span_pii_eval
