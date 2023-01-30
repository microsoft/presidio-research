from collections import Counter

import numpy as np
import pytest

from presidio_evaluator import InputSample, Span

from presidio_evaluator.evaluation import (Evaluator, 
                                            ModelPrediction,
                                            SpanOutput)
from tests.mocks import (
    IdentityTokensMockModel,
    FiftyFiftyIdentityTokensMockModel,
    MockTokensModel,
)


def test_compare_span_simple_case_1():
    annotated_spans =[Span(entity_type = "PER", entity_value = "", start_position = 59, end_position=69),
                      Span(entity_type = "LOC", entity_value = "", start_position = 127, end_position=134),
                      Span(entity_type = "LOC", entity_value = "", start_position = 164, end_position=174),
                      Span(entity_type = "LOC", entity_value = "", start_position = 197, end_position=205),
                      Span(entity_type = "LOC", entity_value = "", start_position = 208, end_position=219),
                      Span(entity_type = "MISC", entity_value = "", start_position = 230, end_position=240)]
    predicted_spans = [Span(entity_type = "PER", entity_value = "", start_position = 24, end_position=30),
                      Span(entity_type = "LOC", entity_value = "", start_position = 124, end_position=134),
                      Span(entity_type = "PER", entity_value = "", start_position = 164, end_position=174),
                      Span(entity_type = "LOC", entity_value = "", start_position = 197, end_position=205),
                      Span(entity_type = "LOC", entity_value = "", start_position = 208, end_position=219),
                      Span(entity_type = "LOC", entity_value = "", start_position = 225, end_position=243)]

    evaluator = Evaluator(entities_to_keep=['PER', 'LOC', 'MISC'])
    span_outputs, evaluation, evaluation_agg_entities_type = evaluator.compare_span(annotated_spans, predicted_spans)

    expected_evaluation = {'strict': {'correct': 2,
                           'incorrect': 3,
                           'partial': 0,
                           'missed': 1,
                           'spurious': 1,
                           'possible': 6,
                           'actual': 6},
                'ent_type': {'correct': 3,
                             'incorrect': 2,
                             'partial': 0,
                             'missed': 1,
                             'spurious': 1,
                             'possible': 6,
                             'actual': 6},
                'partial': {'correct': 3,
                            'incorrect': 0,
                            'partial': 2,
                            'missed': 1,
                            'spurious': 1,
                            'possible': 6,
                            'actual': 6},
                'exact': {'correct': 3,
                          'incorrect': 2,
                          'partial': 0,
                          'missed': 1,
                          'spurious': 1,
                          'possible': 6,
                          'actual': 6}
                }
    print(span_outputs)
    print(expected_evaluation)
    assert evaluation == expected_evaluation

def test_compare_span_strict():
    annotated_spans =[Span(entity_type = "ANIMAL", entity_value = "golden retriever", start_position = 9, end_position=24)]
    predicted_spans = [Span(entity_type = "ANIMAL", entity_value = "golden retriever", start_position = 9, end_position=24)] 

    evaluator = Evaluator(entities_to_keep=["ANIMAL"])
    span_outputs, evaluation, evaluation_agg_entities_type = evaluator.compare_span(annotated_spans, predicted_spans)

    expected_evaluation = {
        'strict': {
            'correct': 1,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        },
        'ent_type': {
            'correct': 1,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        },
        'partial': {
            'correct': 1,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        },
        'exact': {
            'correct': 1,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        }
    }
    expected_span_outputs = [SpanOutput(
                        output_type = "STRICT",
                        predicted_span = Span(entity_type = "ANIMAL", entity_value = "golden retriever", start_position = 9, end_position=24),
                        annotated_span = Span(entity_type = "ANIMAL", entity_value = "golden retriever", start_position = 9, end_position=24),
                        overlap_score = 1)]

    assert len(span_outputs) == len(expected_span_outputs)
    assert all([a.__eq__(b) for a, b in zip(span_outputs, expected_span_outputs)])
    assert evaluation['strict'] == expected_evaluation['strict']
    assert evaluation['ent_type'] == expected_evaluation['ent_type']
    assert evaluation['partial'] == expected_evaluation['partial']
    assert evaluation['exact'] == expected_evaluation['exact']


def test_compare_span_ent_type():
    annotated_spans = [Span(entity_type = "ANIMAL", entity_value = "golden retriever", start_position = 9, end_position=24)] 
    predicted_spans =[Span(entity_type = "ANIMAL", entity_value = "retriever", start_position = 15, end_position=24)]

    evaluator = Evaluator(entities_to_keep=["ANIMAL"])
    span_outputs, evaluation, evaluation_agg_entities_type = evaluator.compare_span(annotated_spans, predicted_spans)

    expected_evaluation = {
        'strict': {
            'correct': 0,
            'incorrect': 1,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        },
        'ent_type': {
            'correct': 1,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        },
        'partial': {
            'correct': 0,
            'incorrect': 0,
            'partial': 1,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual':
            1,
            'possible': 1
        },
        'exact': {
            'correct': 0,
            'incorrect': 1,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        }
    }

    expected_span_outputs = [SpanOutput(
                        output_type = "ENT_TYPE",
                        predicted_span = Span(entity_type = "ANIMAL", entity_value = "retriever", start_position = 15, end_position=24),
                        annotated_span = Span(entity_type = "ANIMAL", entity_value = "golden retriever", start_position = 9, end_position=24),
                        overlap_score = 0.72)]

    assert len(span_outputs) == len(expected_span_outputs)
    assert all([a.__eq__(b) for a, b in zip(span_outputs, expected_span_outputs)])
    assert evaluation['strict'] == expected_evaluation['strict']
    assert evaluation['ent_type'] == expected_evaluation['ent_type']
    assert evaluation['partial'] == expected_evaluation['partial']
    assert evaluation['exact'] == expected_evaluation['exact']

def test_compare_span_exact():
    annotated_spans = [Span(entity_type = "ANIMAL", entity_value = "golden retriever", start_position = 9, end_position=24)] 
    predicted_spans =[Span(entity_type = "SPACESHIP", entity_value = "golden retriever", start_position = 9, end_position=24)]

    evaluator = Evaluator(entities_to_keep=["ANIMAL"])
    span_outputs, evaluation, evaluation_agg_entities_type = evaluator.compare_span(annotated_spans, predicted_spans)

    expected_evaluation = {
        'strict': {
            'correct': 0,
            'incorrect': 1,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        },
        'ent_type': {
            'correct': 0,
            'incorrect': 1,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        },
        'partial': {
            'correct': 1,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        },
        'exact': {
            'correct': 1,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        }
    }

    expected_span_outputs = [SpanOutput(
                        output_type = "EXACT",
                        predicted_span = Span(entity_type = "SPACESHIP", entity_value = "golden retriever", start_position = 9, end_position=24),
                        annotated_span = Span(entity_type = "ANIMAL", entity_value = "golden retriever", start_position = 9, end_position=24),
                        overlap_score = 1)]
    print(span_outputs)

    assert len(span_outputs) == len(expected_span_outputs)
    assert all([a.__eq__(b) for a, b in zip(span_outputs, expected_span_outputs)])
    assert evaluation['strict'] == expected_evaluation['strict']
    assert evaluation['ent_type'] == expected_evaluation['ent_type']
    assert evaluation['partial'] == expected_evaluation['partial']
    assert evaluation['exact'] == expected_evaluation['exact']

def test_compare_span_partial():
    annotated_spans = [Span(entity_type = "ANIMAL", entity_value = "golden retriever", start_position = 9, end_position=24)] 
    predicted_spans =[Span(entity_type = "SPACESHIP", entity_value = "retriever", start_position = 15, end_position=24)]

    evaluator = Evaluator(entities_to_keep=["ANIMAL"])
    span_outputs, evaluation, evaluation_agg_entities_type = evaluator.compare_span(annotated_spans, predicted_spans)

    expected_evaluation = {
        'strict': {
            'correct': 0,
            'incorrect': 1,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        },
        'ent_type': {
            'correct': 0,
            'incorrect': 1,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        },
        'partial': {
            'correct': 0,
            'incorrect': 0,
            'partial': 1,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual':
            1,
            'possible': 1
        },
        'exact': {
            'correct': 0,
            'incorrect': 1,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'precision': 0,
            'recall': 0,
            'actual': 1,
            'possible': 1
        }
    }

    expected_span_outputs = [SpanOutput(
                        output_type = "PARTIAL",
                        predicted_span = Span(entity_type = "SPACESHIP", entity_value = "retriever", start_position = 15, end_position=24),
                        annotated_span = Span(entity_type = "ANIMAL", entity_value = "golden retriever", start_position = 9, end_position=24),
                        overlap_score = 0.72)]
    print(span_outputs)

    assert len(span_outputs) == len(expected_span_outputs)
    assert all([a.__eq__(b) for a, b in zip(span_outputs, expected_span_outputs)])
    assert evaluation['strict'] == expected_evaluation['strict']
    assert evaluation['ent_type'] == expected_evaluation['ent_type']
    assert evaluation['partial'] == expected_evaluation['partial']
    assert evaluation['exact'] == expected_evaluation['exact']

# TODO: refactor those functions
# def test_evaluator_simple():
#     prediction = ["O", "O", "O", "U-ANIMAL"]
#     model = MockTokensModel(prediction=prediction, entities_to_keep=["ANIMAL"])

#     evaluator = Evaluator(model=model)
#     sample = InputSample(
#         full_text="I am the walrus", masked="I am the [ANIMAL]", spans=None
#     )
#     sample.tokens = ["I", "am", "the", "walrus"]
#     sample.tags = ["O", "O", "O", "U-ANIMAL"]

#     evaluated = evaluator.evaluate_sample(sample, prediction)
#     final_evaluation = evaluator.calculate_score([evaluated])

#     assert final_evaluation.pii_precision == 1
#     assert final_evaluation.pii_recall == 1


# def test_evaluate_sample_wrong_entities_to_keep_correct_statistics():
#     prediction = ["O", "O", "O", "U-ANIMAL"]
#     model = MockTokensModel(prediction=prediction)

#     evaluator = Evaluator(model=model, entities_to_keep=["SPACESHIP"])

#     sample = InputSample(
#         full_text="I am the walrus", masked="I am the [ANIMAL]", spans=None
#     )
#     sample.tokens = ["I", "am", "the", "walrus"]
#     sample.tags = ["O", "O", "O", "U-ANIMAL"]

#     evaluated = evaluator.evaluate_sample(sample, prediction)
#     assert evaluated.results[("O", "O")] == 4


# def test_evaluate_same_entity_correct_statistics():
#     prediction = ["O", "U-ANIMAL", "O", "U-ANIMAL"]
#     model = MockTokensModel(prediction=prediction)
#     evaluator = Evaluator(model=model, entities_to_keep=["ANIMAL"])
#     sample = InputSample(
#         full_text="I dog the walrus", masked="I [ANIMAL] the [ANIMAL]", spans=None
#     )
#     sample.tokens = ["I", "am", "the", "walrus"]
#     sample.tags = ["O", "O", "O", "U-ANIMAL"]

#     evaluation_result = evaluator.evaluate_sample(sample, prediction)
#     assert evaluation_result.results[("O", "O")] == 2
#     assert evaluation_result.results[("ANIMAL", "ANIMAL")] == 1
#     assert evaluation_result.results[("O", "ANIMAL")] == 1


# def test_evaluate_multiple_entities_to_keep_correct_statistics():
#     prediction = ["O", "U-ANIMAL", "O", "U-ANIMAL"]
#     entities_to_keep = ["ANIMAL", "PLANT", "SPACESHIP"]
#     model = MockTokensModel(prediction=prediction)
#     evaluator = Evaluator(model=model, entities_to_keep=entities_to_keep)

#     sample = InputSample(
#         full_text="I dog the walrus", masked="I [ANIMAL] the [ANIMAL]", spans=None
#     )
#     sample.tokens = ["I", "am", "the", "walrus"]
#     sample.tags = ["O", "O", "O", "U-ANIMAL"]

#     evaluation_result = evaluator.evaluate_sample(sample, prediction)
#     assert evaluation_result.results[("O", "O")] == 2
#     assert evaluation_result.results[("ANIMAL", "ANIMAL")] == 1
#     assert evaluation_result.results[("O", "ANIMAL")] == 1


# def test_evaluate_multiple_tokens_correct_statistics():
#     prediction = ["O", "O", "O", "B-ANIMAL", "I-ANIMAL", "L-ANIMAL"]
#     model = MockTokensModel(prediction=prediction)
#     evaluator = Evaluator(model=model, entities_to_keep=["ANIMAL"])
#     sample = InputSample(
#         "I am the walrus amaericanus magnifico", masked=None, spans=None
#     )
#     sample.tokens = ["I", "am", "the", "walrus", "americanus", "magnifico"]
#     sample.tags = ["O", "O", "O", "B-ANIMAL", "I-ANIMAL", "L-ANIMAL"]

#     evaluated = evaluator.evaluate_sample(sample, prediction)
#     evaluation = evaluator.calculate_score([evaluated])

#     assert evaluation.pii_precision == 1
#     assert evaluation.pii_recall == 1


# def test_evaluate_multiple_tokens_partial_match_correct_statistics():
#     prediction = ["O", "O", "O", "B-ANIMAL", "L-ANIMAL", "O"]
#     model = MockTokensModel(prediction=prediction)
#     evaluator = Evaluator(model=model, entities_to_keep=["ANIMAL"])
#     sample = InputSample(
#         "I am the walrus amaericanus magnifico", masked=None, spans=None
#     )
#     sample.tokens = ["I", "am", "the", "walrus", "americanus", "magnifico"]
#     sample.tags = ["O", "O", "O", "B-ANIMAL", "I-ANIMAL", "L-ANIMAL"]

#     evaluated = evaluator.evaluate_sample(sample, prediction)
#     evaluation = evaluator.calculate_score([evaluated])

#     assert evaluation.pii_precision == 1
#     assert evaluation.pii_recall == 4 / 6


# def test_evaluate_multiple_tokens_no_match_match_correct_statistics():
#     prediction = ["O", "O", "O", "B-SPACESHIP", "L-SPACESHIP", "O"]
#     model = MockTokensModel(prediction=prediction)
#     evaluator = Evaluator(model=model, entities_to_keep=["ANIMAL"])
#     sample = InputSample(
#         "I am the walrus amaericanus magnifico", masked=None, spans=None
#     )
#     sample.tokens = ["I", "am", "the", "walrus", "americanus", "magnifico"]
#     sample.tags = ["O", "O", "O", "B-ANIMAL", "I-ANIMAL", "L-ANIMAL"]

#     evaluated = evaluator.evaluate_sample(sample, prediction)
#     evaluation = evaluator.calculate_score([evaluated])

#     assert np.isnan(evaluation.pii_precision)
#     assert evaluation.pii_recall == 0


# def test_evaluate_multiple_examples_correct_statistics():
#     prediction = ["U-PERSON", "O", "O", "U-PERSON", "O", "O"]
#     model = MockTokensModel(prediction=prediction)
#     evaluator = Evaluator(model=model, entities_to_keep=["PERSON"])
#     input_sample = InputSample("My name is Raphael or David", masked=None, spans=None)
#     input_sample.tokens = ["My", "name", "is", "Raphael", "or", "David"]
#     input_sample.tags = ["O", "O", "O", "U-PERSON", "O", "U-PERSON"]

#     evaluated = evaluator.evaluate_all(
#         [input_sample, input_sample, input_sample, input_sample]
#     )
#     scores = evaluator.calculate_score(evaluated)
#     assert scores.pii_precision == 0.5
#     assert scores.pii_recall == 0.5


# def test_evaluate_multiple_examples_ignore_entity_correct_statistics():
#     prediction = ["O", "O", "O", "U-PERSON", "O", "U-TENNIS_PLAYER"]
#     model = MockTokensModel(prediction=prediction)

#     evaluator = Evaluator(model=model, entities_to_keep=["PERSON", "TENNIS_PLAYER"])
#     input_sample = InputSample("My name is Raphael or David", masked=None, spans=None)
#     input_sample.tokens = ["My", "name", "is", "Raphael", "or", "David"]
#     input_sample.tags = ["O", "O", "O", "U-PERSON", "O", "U-PERSON"]

#     evaluated = evaluator.evaluate_all(
#         [input_sample, input_sample, input_sample, input_sample]
#     )
#     scores = evaluator.calculate_score(evaluated)
#     assert scores.pii_precision == 1
#     assert scores.pii_recall == 1


# def test_confusion_matrix_correct_metrics():
#     from collections import Counter

#     evaluated = [
#         EvaluationResult(
#             results=Counter(
#                 {
#                     ("O", "O"): 150,
#                     ("O", "PERSON"): 30,
#                     ("O", "COMPANY"): 30,
#                     ("PERSON", "PERSON"): 40,
#                     ("COMPANY", "COMPANY"): 40,
#                     ("PERSON", "COMPANY"): 10,
#                     ("COMPANY", "PERSON"): 10,
#                     ("PERSON", "O"): 30,
#                     ("COMPANY", "O"): 30,
#                 }
#             ),
#             model_errors=None,
#             text=None,
#         )
#     ]

#     model = MockTokensModel(prediction=None)
#     evaluator = Evaluator(model=model, entities_to_keep=["PERSON", "COMPANY"])
#     scores = evaluator.calculate_score(evaluated, beta=2.5)

#     assert scores.pii_precision == 0.625
#     assert scores.pii_recall == 0.625
#     assert scores.entity_recall_dict["PERSON"] == 0.5
#     assert scores.entity_precision_dict["PERSON"] == 0.5
#     assert scores.entity_recall_dict["COMPANY"] == 0.5
#     assert scores.entity_precision_dict["COMPANY"] == 0.5


# def test_confusion_matrix_2_correct_metrics():
#     from collections import Counter

#     evaluated = [
#         EvaluationResult(
#             results=Counter(
#                 {
#                     ("O", "O"): 65467,
#                     ("O", "ORG"): 4189,
#                     ("GPE", "O"): 3370,
#                     ("PERSON", "PERSON"): 2024,
#                     ("GPE", "PERSON"): 1488,
#                     ("GPE", "GPE"): 1033,
#                     ("O", "GPE"): 964,
#                     ("ORG", "ORG"): 914,
#                     ("O", "PERSON"): 834,
#                     ("GPE", "ORG"): 401,
#                     ("PERSON", "ORG"): 35,
#                     ("PERSON", "O"): 33,
#                     ("ORG", "O"): 8,
#                     ("PERSON", "GPE"): 5,
#                     ("ORG", "PERSON"): 1,
#                 }
#             ),
#             model_errors=None,
#             text=None,
#         )
#     ]

#     model = MockTokensModel(prediction=None)
#     evaluator = Evaluator(model=model)
#     scores = evaluator.calculate_score(evaluated, beta=2.5)

#     pii_tp = (
#         evaluated[0].results[("PERSON", "PERSON")]
#         + evaluated[0].results[("ORG", "ORG")]
#         + evaluated[0].results[("GPE", "GPE")]
#         + evaluated[0].results[("ORG", "GPE")]
#         + evaluated[0].results[("ORG", "PERSON")]
#         + evaluated[0].results[("GPE", "ORG")]
#         + evaluated[0].results[("GPE", "PERSON")]
#         + evaluated[0].results[("PERSON", "GPE")]
#         + evaluated[0].results[("PERSON", "ORG")]
#     )

#     pii_fp = (
#         evaluated[0].results[("O", "PERSON")]
#         + evaluated[0].results[("O", "GPE")]
#         + evaluated[0].results[("O", "ORG")]
#     )

#     pii_fn = (
#         evaluated[0].results[("PERSON", "O")]
#         + evaluated[0].results[("GPE", "O")]
#         + evaluated[0].results[("ORG", "O")]
#     )

#     assert scores.pii_precision == pii_tp / (pii_tp + pii_fp)
#     assert scores.pii_recall == pii_tp / (pii_tp + pii_fn)


# def test_dataset_to_metric_identity_model():
#     import os

#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     input_samples = InputSample.read_dataset_json(
#         "{}/data/generated_small.json".format(dir_path), length=10
#     )

#     model = IdentityTokensMockModel()
#     evaluator = Evaluator(model=model)
#     evaluation_results = evaluator.evaluate_all(input_samples)
#     metrics = evaluator.calculate_score(evaluation_results)

#     assert metrics.pii_precision == 1
#     assert metrics.pii_recall == 1


# def test_dataset_to_metric_50_50_model():
#     import os

#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     input_samples = InputSample.read_dataset_json(
#         "{}/data/generated_small.json".format(dir_path), length=100
#     )

#     # Replace 50% of the predictions with a list of "O"
#     model = FiftyFiftyIdentityTokensMockModel()
#     evaluator = Evaluator(model=model, entities_to_keep=["PERSON"])
#     evaluation_results = evaluator.evaluate_all(input_samples)
#     metrics = evaluator.calculate_score(evaluation_results)

#     print(metrics.pii_precision)
#     print(metrics.pii_recall)
#     print(metrics.pii_f)

#     assert metrics.pii_precision == 1
#     assert metrics.pii_recall < 0.75
#     assert metrics.pii_recall > 0.25


# def test_align_entity_types_correct_output():

#     sample1 = InputSample(
#         "I live in ABC",
#         spans=[Span("A", "a", 0, 1), Span("A", "a", 10, 11), Span("B", "b", 100, 101)],
#         create_tags_from_span=False,
#     )
#     sample2 = InputSample(
#         "I live in ABC",
#         spans=[Span("A", "a", 0, 1), Span("A", "a", 10, 11), Span("C", "c", 100, 101)],
#         create_tags_from_span=False,
#     )
#     samples = [sample1, sample2]
#     mapping = {
#         "A": "1",
#         "B": "2",
#         "C": "1",
#     }

#     new_samples = Evaluator.align_entity_types(samples, mapping)

#     count_per_entity = Counter()
#     for sample in new_samples:
#         for span in sample.spans:
#             count_per_entity[span.entity_type] += 1

#     assert count_per_entity["1"] == 5
#     assert count_per_entity["2"] == 1


# def test_align_entity_types_wrong_mapping_exception():

#     sample1 = InputSample(
#         "I live in ABC",
#         spans=[Span("A", "a", 0, 1), Span("A", "a", 10, 11), Span("B", "b", 100, 101)],
#         create_tags_from_span=False,
#     )

#     entities_mapping = {"Z": "z"}

#     with pytest.raises(ValueError):
#         Evaluator.align_entity_types(
#             input_samples=[sample1], entities_mapping=entities_mapping
#         )
