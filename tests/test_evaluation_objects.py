from spacy.tokens import Token
import pytest

from presidio_evaluator.evaluation import TokenOutput


@pytest.fixture(params=[[TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "PERSON", token = "Network"),
                        TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "LOCATION", token = "Network"),
                        TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "LOCATION", token = "str"),
                        TokenOutput(error_type = "FN", annotated_tag = "PHONE_NUMBER", predicted_tag = "O", token = "60"),
                        TokenOutput(error_type = "FN", annotated_tag = "PHONE_NUMBER", predicted_tag = "O", token = "80"),
                        TokenOutput(error_type = "FN", annotated_tag = "PHONE_NUMBER", predicted_tag = "O", token = "90")]])
def get_token_errors_input(request):
    return request.param

def test_get_fp_token_error(get_token_errors_input, error_type="FP"):
    # Get all false positive errors
    all_fp_errors = TokenOutput.get_token_error_by_type(errors=get_token_errors_input, error_type=error_type)
    # Expected output
    all_fp_errors_expected = [TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "PERSON", token = "Network"),
                              TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "LOCATION", token = "Network"),
                              TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "LOCATION", token = "str")]
    assert len(all_fp_errors) == len(all_fp_errors_expected)
    assert all([a.__eq__(b) for a, b in zip(all_fp_errors, all_fp_errors_expected)])

def test_get_fn_token_error(get_token_errors_input, error_type="FN"):
    # Get all false negative errors
    all_fn_errors = TokenOutput.get_token_error_by_type(errors=get_token_errors_input, error_type=error_type)
    print(all_fn_errors)
    # Expected output
    all_fn_errors_expected = [TokenOutput(error_type = "FN", annotated_tag = "PHONE_NUMBER", predicted_tag = "O", token = "60"),
                              TokenOutput(error_type = "FN", annotated_tag = "PHONE_NUMBER", predicted_tag = "O", token = "80"),
                              TokenOutput(error_type = "FN", annotated_tag = "PHONE_NUMBER", predicted_tag = "O", token = "90")]
    assert len(all_fn_errors) == len(all_fn_errors_expected)
    assert all([a.__eq__(b) for a, b in zip(all_fn_errors, all_fn_errors_expected)])

def test_get_fp_token_error_by_entity(get_token_errors_input, error_type="FP", entity=['LOCATION']):
    # Get all false negative errors in LOCATION entity
    fp_errors_by_entity = TokenOutput.get_token_error_by_type(errors=get_token_errors_input, error_type=error_type, entity=entity)
    # Expected output
    fp_errors_by_entity_expected = [TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "LOCATION", token = "Network"),
                                    TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "LOCATION", token = "str")]
    assert len(fp_errors_by_entity) == len(fp_errors_by_entity_expected)
    assert all([a.__eq__(b) for a, b in zip(fp_errors_by_entity, fp_errors_by_entity_expected)])

def test_get_most_fp_token_error_by_entity(get_token_errors_input, error_type="FP", entity=['LOCATION'], n=1):
    # Get top 1 false negative errors in LOCATION entity
    top1_fp_errors_by_entity = TokenOutput.get_token_error_by_type(errors=get_token_errors_input, error_type=error_type, entity=entity, n=n)
    # Expected output
    top1_fp_errors_by_entity_expected = [TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "LOCATION", token = "Network")]
    assert len(top1_fp_errors_by_entity) == len(top1_fp_errors_by_entity_expected)
    assert all([a.__eq__(b) for a, b in zip(top1_fp_errors_by_entity, top1_fp_errors_by_entity_expected)])

def test_get_most_common_token_by_type(get_token_errors_input, n=1):
    # Get the most common tokens errors 
    top1_errors = TokenOutput.get_common_token(errors=get_token_errors_input, n=n)
    # Expected output
    top1_errors_expected = [TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "PERSON", token = "Network"),
                            TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "LOCATION", token = "Network")]
    assert len(top1_errors) == len(top1_errors_expected)
    assert all([a.__eq__(b) for a, b in zip(top1_errors, top1_errors_expected)])


@pytest.mark.parametrize(
    "token_output1, token_output2, expected_output",
    [
        (TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "PERSON", token = "Network"), 
                     TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "PERSON", token = "Network"), True),
        (TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "PERSON", token = "Network"), 
                     TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "LOCATION", token = "Network"), False),
        (TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "PERSON", token = "Network"), 
                     TokenOutput(error_type = "FP", annotated_tag = "LOCATION", predicted_tag = "PERSON", token = "Network"), False),
        (TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "PERSON", token = "Network"), 
                     TokenOutput(error_type = "FP", annotated_tag = "O", predicted_tag = "PERSON", token = "Str"), False),
    ],
)
def test_eq_token_output(
    token_output1, token_output2, expected_output
):

    is_eq = token_output1.__eq__(token_output2)
    assert is_eq == expected_output

