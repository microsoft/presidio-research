# """
# Test cases for SpanEvaluator to verify correct operation of the entity evaluation pipeline.
# Tests focus on:
# 1. Span creation and normalization
# 2. Merging of adjacent spans
# 3. IoU calculation
# 4. Matching logic
# 5. Error analysis and result population
# 6. Integration with visualization components
# """
#
# import numpy as np
# import pytest
# import pandas as pd
# from presidio_evaluator.data_objects import Span
# from presidio_evaluator.evaluation.span_evaluator import SpanEvaluator
# from presidio_evaluator.evaluation.evaluation_result import EvaluationResult
# from presidio_evaluator.evaluation import ErrorType
# from tests.mocks import MockModel
#
#
# # ===== Fixtures =====
#
#
# @pytest.fixture
# def sample_df():
#     """Create a sample DataFrame for testing span creation and normalization."""
#     return pd.DataFrame(
#         {
#             "sentence_id": [0, 0, 0, 0, 0],
#             "token": ["John", "Smith", "lives", "in", "London"],
#             "annotation": ["PERSON", "PERSON", "O", "O", "LOCATION"],
#             "prediction": ["PERSON", "PERSON", "O", "O", "LOCATION"],
#             "start_indices": [0, 5, 11, 17, 20],
#         }
#     )
#
#
# @pytest.fixture
# def sample_df_with_skipwords():
#     """Create a sample DataFrame with skip words for testing adjacency and merging."""
#     return pd.DataFrame(
#         {
#             "sentence_id": [5, 5, 5, 5, 5, 5, 5],
#             "token": ["Dr", ",", "Jane", "Smith", "-", "MD", "arrived"],
#             "annotation": ["PERSON", "O", "PERSON", "PERSON", "O", "PERSON", "O"],
#             "prediction": ["PERSON", "O", "PERSON", "PERSON", "O", "PERSON", "O"],
#             "start_indices": [0, 4, 6, 11, 16, 18, 21],
#         }
#     )
#
#
# @pytest.fixture
# def mock_span_evaluator():
#     """Create a SpanEvaluator instance for testing."""
#     return SpanEvaluator(model=MockModel(), iou_threshold=0.5, char_based=True)
#
#
# # ===== Core End-to-End Evaluation Tests =====
#
#
# @pytest.mark.parametrize(
#     "annotation, prediction, tokens, start_indices, TP, num_annotated, num_predicted, char_based, expected_per_type",
#     [
#         # BASIC ENTITY MATCHING
#         (
#             ["PERSON", "O", "O", "O"],
#             ["PERSON", "O", "O", "O"],
#             ["David", "is", "my", "friend"],
#             [0, 6, 9, 12],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["EMAIL", "O", "O", "O"],
#             ["EMAIL", "O", "O", "O"],
#             ["user@example.com", "sent", "a", "message"],
#             [0, 17, 22, 24],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "EMAIL": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         # SKIP WORD HANDLING
#         (
#             ["PERSON", "PERSON", "PERSON", "O"],
#             ["PERSON", "O", "PERSON", "O"],
#             ["David", "is", "living", "abroad"],
#             [0, 6, 9, 16],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "O", "PERSON", "O"],
#             ["PERSON", "O", "PERSON", "O"],
#             ["John", "and", "Mary", "came"],
#             [0, 5, 9, 14],
#             1,
#             1,
#             1,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "O", "O", "O", "PERSON", "O"],
#             ["PERSON", "O", "O", "O", "PERSON", "O"],
#             ["John", "and", "the", "other", "Smith", "arrived"],
#             [0, 5, 9, 13, 19, 25],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         # PUNCTUATION HANDLING
#         (
#             ["PERSON", "O", "PERSON", "PERSON"],
#             ["PERSON", "PERSON", "PERSON", "PERSON"],
#             ["David", ",", "Maxwell", "Morris"],
#             [0, 6, 8, 16],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "O", "O", "PERSON"],
#             ["PERSON", "O", "O", "PERSON"],
#             ["David", "-", "-", "Morris"],
#             [0, 6, 8, 10],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "O", "PERSON", "O"],
#             ["PERSON", "O", "PERSON", "O"],
#             ["Dr.", ",", "Smith", "arrived"],
#             [0, 4, 6, 12],
#             1,
#             1,
#             1,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "O", "O", "O", "PERSON"],
#             ["PERSON", "O", "O", "O", "PERSON"],
#             ["James", ".", "-", "/", "Bond"],
#             [0, 6, 8, 10, 12],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         # BOUNDARY MISMATCHES
#         (
#             ["LOCATION", "LOCATION", "LOCATION", "O"],
#             ["O", "LOCATION", "LOCATION", "O"],
#             ["New", "York", "City", "is"],
#             [0, 4, 9, 14],
#             0,
#             1,
#             0, # Not counted as a prediction (Scenario 5A)
#             False,
#             {
#                 "LOCATION": {
#                     "tp": 0,
#                     "fp": 0, # Not considered FP (Scenario 5A)
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 0,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         (
#             ["O", "O", "PERSON", "PERSON"],
#             ["O", "PERSON", "O", "PERSON"],
#             ["I", "met", "John", "Doe"],
#             [0, 2, 6, 11],
#             0,
#             1,
#             2,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 2,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 2,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "PERSON", "O"],
#             ["PERSON", "PERSON", "PERSON", "PERSON"],
#             ["Anna", "Marie", "Smith", "Loves"],
#             [0, 5, 11, 17],
#             0,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "PERSON"],
#             ["PERSON", "PERSON", "O"],
#             ["John", "Doe", "Smith"],
#             [0, 5, 9],
#             0,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         (
#             ["O", "PERSON", "PERSON", "PERSON", "O"],
#             ["PERSON", "PERSON", "PERSON", "O", "O"],
#             ["Mr", "John", "Middle", "Smith", "Jr"],
#             [0, 3, 8, 15, 21],
#             0,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         # ENTITY TYPE MISMATCHES
#         (
#             ["PERSON", "O", "O", "O"],
#             ["LOCATION", "O", "O", "O"],
#             ["Paris", "is", "beautiful", "today"],
#             [0, 6, 9, 19],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 0,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 0,
#                     "precision": np.nan,
#                     "recall": 0.0,
#                 },
#                 "LOCATION": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 0,
#                     "num_annotated": 0,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": np.nan,
#                 },
#             },
#         ),
#         (
#             ["CREDIT_CARD", "CREDIT_CARD", "CREDIT_CARD", "CREDIT_CARD"],
#             ["PHONE_NUMBER", "PHONE_NUMBER", "PHONE_NUMBER", "PHONE_NUMBER"],
#             ["1234", "5678", "9012", "3456"],
#             [0, 5, 10, 15],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "CREDIT_CARD": {
#                     "tp": 0,
#                     "fp": 0,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 0,
#                     "precision": np.nan,
#                     "recall": 0.0,
#                 },
#                 "PHONE_NUMBER": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 0,
#                     "num_annotated": 0,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": np.nan,
#                 },
#             },
#         ),
#         # MULTIPLE ENTITY SCENARIOS
#         (
#             ["PERSON", "O", "O", "LOCATION"],
#             ["PERSON", "O", "PERSON", "PERSON"],
#             ["Alice", "went", "to", "Paris"],
#             [0, 6, 11, 14],
#             2,
#             2,
#             2,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 1,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 2,
#                     "precision": 0.5,
#                     "recall": 1.0,
#                 },
#                 "LOCATION": {
#                     "tp": 0,
#                     "fp": 0,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 0,
#                     "precision": np.nan,
#                     "recall": 0.0,
#                 },
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
#             ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
#             ["Barack", "Obama", "visited", "New", "York"],
#             [0, 7, 13, 21, 25],
#             2,
#             2,
#             2,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#                 "LOCATION": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
#             ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
#             ["John", "Doe", "and", "Jane", "Smith"],
#             [0, 5, 9, 13, 18],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
#             ["PERSON", "PERSON", "PERSON", "PERSON", "PERSON"],
#             ["John", "Doe", "and", "Jane", "Smith"],
#             [0, 5, 9, 13, 18],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "O", "PERSON", "O", "LOCATION"],
#             ["PERSON", "O", "PERSON", "O", "LOCATION"],
#             ["John", "met", "Jane", "in", "Paris"],
#             [0, 5, 9, 14, 17],
#             2,
#             2,
#             2,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 2,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 2,
#                     "num_predicted": 2,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#                 "LOCATION": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "O", "LOCATION", "O", "DATE", "DATE"],
#             ["PERSON", "PERSON", "O", "ORGANIZATION", "O", "DATE", "O"],
#             ["John", "Smith", "visited", "London", "on", "January", "1st"],
#             [0, 5, 11, 19, 26, 29, 37],
#             1,
#             2,
#             2,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#                 "LOCATION": {
#                     "tp": 0,
#                     "fp": 0,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 0,
#                     "precision": np.nan,
#                     "recall": 0.0,
#                 },
#                 "DATE": {
#                     "tp": 0,
#                     "fp": 0,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#                 "ORGANIZATION": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 0,
#                     "num_annotated": 0,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": np.nan,
#                 },
#             },
#         ),
#         # OVERLAPPING/NESTED ENTITIES
#         (
#             ["PERSON", "PERSON", "PERSON", "PERSON", "O"],
#             ["PERSON", "O", "PERSON", "PERSON", "O"],
#             ["Sir", "Arthur", "Conan", "Doyle", "wrote"],
#             [0, 4, 11, 17, 23],
#             0,
#             1,
#             2,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 2,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 2,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "PERSON", "O"],
#             ["PERSON", "O", "PERSON", "PERSON"],
#             ["James", "Robert", "Smith", "III"],
#             [0, 6, 13, 19],
#             0,
#             1,
#             2,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 2,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 2,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         # SPECIAL CHARACTERS AND FORMATTING
#         (
#             ["PERSON", "PERSON", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
#             ["PERSON", "PERSON", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
#             ["O'Brien", "Jr.", "at", "McDonald's", "Corp.", "Inc."],
#             [0, 8, 12, 15, 27, 34],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#                 "ORGANIZATION": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#             },
#         ),
#         (
#             ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
#             ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION"],
#             ["United", "-", "States", "Government"],
#             [0, 7, 9, 16],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "ORGANIZATION": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "O", "O"],
#             ["PERSON", "PERSON", "O", "O"],
#             ["José", "Martínez", "from", "España"],
#             [0, 5, 14, 19],
#             1,
#             1,
#             1,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["ID_NUMBER", "ID_NUMBER", "ID_NUMBER"],
#             ["ID_NUMBER", "ID_NUMBER", "ID_NUMBER"],
#             ["ID", "-", "12345"],
#             [0, 3, 5],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "ID_NUMBER": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         # EDGE CASES
#         (
#             ["O"] * 1000,
#             ["O"] * 1000,
#             ["word"] * 1000,
#             list(range(0, 5000, 5)),
#             0,
#             0,
#             0,
#             False,
#             {},
#         ),
#         (
#             ["O", "O", "O", "O"],
#             ["PERSON", "O", "LOCATION", "O"],
#             ["This", "is", "London", "now"],
#             [0, 5, 8, 15],
#             0,
#             0,
#             1,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 0,
#                     "num_annotated": 0,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": np.nan,
#                 },
#                 "LOCATION": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 0,
#                     "num_annotated": 0,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": np.nan,
#                 },
#             },
#         ),
#         (
#             ["PERSON", "O", "LOCATION", "LOCATION"],
#             ["O", "O", "O", "O"],
#             ["Emma", "travels", "to", "Berlin"],
#             [0, 5, 13, 16],
#             0,
#             2,
#             0,
#             False,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 0,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 0,
#                     "precision": np.nan,
#                     "recall": 0.0,
#                 },
#                 "LOCATION": {
#                     "tp": 0,
#                     "fp": 0,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 0,
#                     "precision": np.nan,
#                     "recall": 0.0,
#                 },
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "O", "O"],
#             ["O", "O", "O", "O"],
#             ["John", "Smith", "works", "here"],
#             [0, 5, 11, 17],
#             0,
#             1,
#             0,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 0,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 0,
#                     "precision": np.nan,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         (
#             ["O", "O", "O", "O"],
#             ["PERSON", "PERSON", "O", "O"],
#             ["John", "Smith", "works", "here"],
#             [0, 5, 11, 17],
#             0,
#             0,
#             1,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 0,
#                     "num_annotated": 0,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": np.nan,
#                 }
#             },
#         ),
#         (
#             ["LOCATION", "O", "LOCATION", "O", "LOCATION"],
#             ["LOCATION", "O", "LOCATION", "O", "LOCATION"],
#             ["UK", "and", "US", "and", "EU"],
#             [0, 3, 7, 10, 14],
#             1,
#             1,
#             1,
#             False,
#             {
#                 "LOCATION": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         # CHARACTER-BASED EVALUATION
#         (
#             ["PERSON", "PERSON", "O", "O"],
#             ["PERSON", "PERSON", "O", "O"],
#             ["John", "Smith", "works", "here"],
#             [0, 5, 11, 17],
#             1,
#             1,
#             1,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
#             ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
#             ["John", "Smith", "visited", "New", "York"],
#             [0, 5, 11, 19, 23],
#             2,
#             2,
#             2,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#                 "LOCATION": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "O", "O"],
#             ["PERSON", "O", "O", "O"],
#             ["John", "Smith", "is", "here"],
#             [0, 5, 11, 14],
#             0,
#             1,
#             1,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "PERSON", "PERSON"],
#             ["PERSON", "PERSON", "PERSON", "PERSON"],
#             ["John", "F.", "Kennedy", "Jr."],
#             [0, 5, 8, 16],
#             1,
#             1,
#             1,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "O", "O"],
#             ["PERSON", "PERSON", "PERSON", "O"],
#             ["John", "Smith", "Jr.", "arrived"],
#             [0, 5, 11, 15],
#             0,
#             1,
#             1,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "PERSON", "O"],
#             ["PERSON", "PERSON", "O", "O"],
#             ["John", "F.", "Kennedy", "arrived"],
#             [0, 5, 8, 16],
#             0,
#             1,
#             1,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "O", "ORGANIZATION"],
#             ["PERSON", "O", "PERSON", "ORGANIZATION"],
#             ["John", "Smith", "at", "Microsoft"],
#             [0, 5, 11, 14],
#             1,
#             2,
#             2,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 },
#                 "ORGANIZATION": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 },
#             },
#         ),
#         (
#             ["ADDRESS", "ADDRESS", "ADDRESS", "ADDRESS", "ADDRESS", "O"],
#             ["ADDRESS", "O", "ADDRESS", "ADDRESS", "ADDRESS", "O"],
#             ["123", "Main", "Street", "Suite", "100", "is"],
#             [0, 4, 9, 16, 22, 26],
#             1,
#             1,
#             1,
#             True,
#             {
#                 "ADDRESS": {
#                     "tp": 1,
#                     "fp": 0,
#                     "fn": 0,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 1.0,
#                     "recall": 1.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "PERSON", "PERSON", "O"],
#             ["PERSON", "PERSON", "O", "PERSON", "O"],
#             ["Dr.", "John", "F.", "Kennedy", "spoke"],
#             [0, 4, 9, 12, 20],
#             0,
#             1,
#             2,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 0,
#                     "fn": 2,
#                     "num_annotated": 1,
#                     "num_predicted": 2,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 }
#             },
#         ),
#         (
#             ["PERSON", "PERSON", "PERSON", "O"],
#             ["TITLE", "PERSON", "LOCATION", "O"],
#             ["Dr.", "John", "London", "arrived"],
#             [0, 4, 9, 15],
#             1,
#             1,
#             1,
#             True,
#             {
#                 "PERSON": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 1,
#                     "num_annotated": 1,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": 0.0,
#                 },
#                 "TITLE": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 0,
#                     "num_annotated": 0,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": np.nan,
#                 },
#                 "LOCATION": {
#                     "tp": 0,
#                     "fp": 1,
#                     "fn": 0,
#                     "num_annotated": 0,
#                     "num_predicted": 1,
#                     "precision": 0.0,
#                     "recall": np.nan,
#                 },
#             },
#         ),
#     ],
# )
# def test_evaluate(
#     annotation,
#     prediction,
#     tokens,
#     start_indices,
#     TP,
#     num_annotated,
#     num_predicted,
#     char_based,
#     expected_per_type,
# ):
#     """Test end-to-end evaluation with various input scenarios."""
#     # Build the DataFrame expected by SpanEvaluator
#     df = pd.DataFrame(
#         {
#             "sentence_id": [0] * len(tokens),
#             "token": tokens,
#             "annotation": annotation,
#             "prediction": prediction,
#             "start_indices": start_indices,
#         }
#     )
#
#     evaluator = SpanEvaluator(
#         model=MockModel(),
#         iou_threshold=0.9,
#         char_based=char_based,
#     )
#     result = EvaluationResult()
#     result = evaluator.calculate_score_on_df(
#         per_type=True, results_df=df, evaluation_result=result
#     )
#     global_pii_df = evaluator.create_global_entities_df(results_df=df)
#     result = evaluator.calculate_score_on_df(
#         per_type=False, results_df=global_pii_df, evaluation_result=result
#     )
#
#     # Calculate expected metrics
#     expected_recall = TP / num_annotated if num_annotated > 0 else np.nan
#     expected_precision = TP / num_predicted if num_predicted > 0 else np.nan
#
#     # Assert that the result is an EvaluationResult instance
#     assert isinstance(result, EvaluationResult)
#
#     # Check counts
#     assert result.pii_predicted == num_predicted
#     assert result.pii_annotated == num_annotated
#     assert result.pii_true_positives == TP
#
#     # Handle nan values separately from numeric values
#     if np.isnan(expected_precision):
#         assert np.isnan(
#             result.pii_precision
#         ), f"Expected NaN precision, got {result.pii_precision}"
#     else:
#         assert (
#             result.pii_precision == pytest.approx(expected_precision)
#         ), f"Precision mismatch: expected {expected_precision}, got {result.pii_precision}"
#
#     if np.isnan(expected_recall):
#         assert np.isnan(
#             result.pii_recall
#         ), f"Expected NaN recall, got {result.pii_recall}"
#     else:
#         assert result.pii_recall == pytest.approx(
#             expected_recall
#         ), f"Recall mismatch: expected {expected_recall}, got {result.pii_recall}"
#
#     # NEW: Validate per-type metrics using expected values
#     if expected_per_type:
#         for entity_type, expected_metrics in expected_per_type.items():
#             assert (
#                 entity_type in result.per_type
#             ), f"Entity type {entity_type} missing from per_type results"
#
#             per_type_metrics = result.per_type[entity_type]
#
#             # Validate counts
#             if "tp" in expected_metrics:
#                 assert (
#                     per_type_metrics.true_positives == expected_metrics["tp"]
#                 ), f"Expected {expected_metrics['tp']} true positives for {entity_type}, got {per_type_metrics.true_positives}"
#
#             if "fp" in expected_metrics:
#                 assert (
#                     per_type_metrics.false_positives == expected_metrics["fp"]
#                 ), f"Expected {expected_metrics['fp']} false positives for {entity_type}, got {per_type_metrics.false_positives}"
#
#             if "fn" in expected_metrics:
#                 assert (
#                     per_type_metrics.false_negatives == expected_metrics["fn"]
#                 ), f"Expected {expected_metrics['fn']} false negatives for {entity_type}, got {per_type_metrics.false_negatives}"
#
#             if "num_annotated" in expected_metrics:
#                 assert (
#                     per_type_metrics.num_annotated == expected_metrics["num_annotated"]
#                 ), f"Expected {expected_metrics['num_annotated']} num_annotated for {entity_type}, got {per_type_metrics.num_annotated}"
#
#             if "num_predicted" in expected_metrics:
#                 assert (
#                     per_type_metrics.num_predicted == expected_metrics["num_predicted"]
#                 ), f"Expected {expected_metrics['num_predicted']} num_predicted for {entity_type}, got {per_type_metrics.num_predicted}"
#
#             # Validate computed metrics
#             if "precision" in expected_metrics:
#                 if np.isnan(expected_metrics["precision"]):
#                     assert np.isnan(
#                         per_type_metrics.precision
#                     ), f"Expected NaN precision for {entity_type}"
#                 else:
#                     assert (
#                         per_type_metrics.precision
#                         == pytest.approx(expected_metrics["precision"])
#                     ), f"Expected precision {expected_metrics['precision']} for {entity_type}, got {per_type_metrics.precision}"
#
#             if "recall" in expected_metrics:
#                 if np.isnan(expected_metrics["recall"]):
#                     assert np.isnan(
#                         per_type_metrics.recall
#                     ), f"Expected NaN recall for {entity_type}"
#                 else:
#                     assert (
#                         per_type_metrics.recall
#                         == pytest.approx(expected_metrics["recall"])
#                     ), f"Expected recall {expected_metrics['recall']} for {entity_type}, got {per_type_metrics.recall}"
#
#     # Also validate that we don't have unexpected entity types
#     for entity_type in result.per_type:
#         if expected_per_type:
#             assert (
#                 entity_type in expected_per_type
#             ), f"Unexpected entity type {entity_type} found in results"
#
#
# def test_evaluate_with_custom_skipwords():
#     df = pd.DataFrame(
#         {
#             "sentence_id": [0] * 5,
#             "token": ["David", "paid", "the", "bill", "today"],
#             "annotation": ["PERSON", "O", "O", "O", "O"],
#             "prediction": ["PERSON", "O", "O", "PERSON", "O"],
#             "start_indices": [0, 6, 11, 15, 20],
#         }
#     )
#
#     evaluator = SpanEvaluator(model=MockModel(), iou_threshold=0.9, skip_words=["bill"])
#     result = EvaluationResult()
#     result = evaluator.calculate_score_on_df(
#         per_type=False, results_df=df, evaluation_result=result
#     )
#     global_pii_df = evaluator.create_global_entities_df(results_df=df)
#     result = evaluator.calculate_score_on_df(
#         per_type=False, results_df=global_pii_df, evaluation_result=result
#     )
#
#     # Calculate expected metrics
#     expected_recall = 1
#     expected_precision = 1
#
#     # Assert that the result is an EvaluationResult instance
#     assert isinstance(result, EvaluationResult)
#
#     assert result.pii_recall == pytest.approx(
#         expected_recall
#     ), f"Recall mismatch: expected {expected_recall}, got {result.pii_recall}"
#     assert result.pii_precision == pytest.approx(
#         expected_precision
#     ), f"Precision mismatch: expected {expected_precision}, got {result.pii_precision}"
#
#
# # ===== Component-Level Unit Tests =====
#
#
# def test_normalize_tokens(mock_span_evaluator):
#     """Test that token normalization correctly handles skip words."""
#
#     tokens = ["Dr", ",", "Jane", "Smith", "-", "MD"]
#     start_indices = [0, 4, 6, 11, 16, 18]
#
#     normalized_tokens, normalized_indices = mock_span_evaluator._normalize_tokens(
#         tokens, start_indices
#     )
#
#     # Skip words should be removed
#     assert normalized_tokens == ["dr", "jane", "smith", "md"]
#     assert normalized_indices == [0, 6, 11, 18]
#
#
# def test_create_spans(mock_span_evaluator, sample_df):
#     """Test that spans are correctly created from tokens with proper normalization indices."""
#     spans = mock_span_evaluator._create_spans(sample_df, "annotation")
#
#     # Should create two spans: PERSON and LOCATION
#     assert len(spans) == 2
#
#     # Check the PERSON span details
#     person_span = spans[0]
#     assert person_span.entity_type == "PERSON"
#     assert person_span.entity_value == "John Smith"
#     assert person_span.start_position == 0
#     assert person_span.end_position == 10  # 5 (start of Smith) + 5 (length of Smith)
#     assert person_span.normalized_tokens == ["john", "smith"]
#     assert person_span.normalized_start_index == 0
#     assert person_span.normalized_end_index == 10
#
#     # Check the LOCATION span details
#     location_span = spans[1]
#     assert location_span.entity_type == "LOCATION"
#     assert location_span.entity_value == "London"
#     assert location_span.start_position == 20
#     assert location_span.end_position == 26  # 20 + 6
#     assert location_span.normalized_start_index == 20
#     assert location_span.normalized_end_index == 26
#
#
# def test_are_spans_adjacent(mock_span_evaluator, sample_df_with_skipwords):
#     """Test that span adjacency is correctly identified when separated by skip words."""
#     spans = mock_span_evaluator._create_spans(sample_df_with_skipwords, "annotation")
#
#     # Get the spans representing "Dr." and "Jane Smith"
#     dr_span = spans[0]
#     jane_smith_span = spans[1]
#
#     # Test that spans separated by a comma (skip word) are adjacent
#     assert mock_span_evaluator._are_spans_adjacent(
#         dr_span, jane_smith_span, sample_df_with_skipwords
#     )
#
#
# def test_merge_adjacent_spans(mock_span_evaluator, sample_df_with_skipwords):
#     """Test that adjacent spans of the same entity type are merged correctly."""
#
#     spans = mock_span_evaluator._create_spans(sample_df_with_skipwords, "annotation")
#
#     # Before merging, we should have 1 PERSON spans, the others are only skip words
#     assert len(spans) == 3
#     assert [span.entity_type for span in spans] == ["PERSON", "PERSON", "PERSON"]
#
#     # Merge adjacent spans
#     merged_spans = mock_span_evaluator._merge_adjacent_spans(
#         spans, sample_df_with_skipwords
#     )
#
#     # After merging, we should have 1 PERSON spans: Dr. Jane Smith MD
#     assert len(merged_spans) == 1
#     merged_span = merged_spans[0]
#     assert merged_span.entity_type == "PERSON"
#     assert merged_span.entity_value == "Dr Jane Smith MD"
#
#     # Check normalized indices are updated correctly after merging
#     assert merged_span.normalized_start_index == min(
#         [span.normalized_start_index for span in spans]
#     )
#     assert merged_span.normalized_end_index == max(
#         [span.normalized_end_index for span in spans]
#     )
#
#
# def test_calculate_iou_token_based():
#     """Test IoU calculation with token-based evaluation."""
#     span1 = Span(
#         entity_type="PERSON",
#         entity_value="John Smith",
#         start_position=0,
#         end_position=10,
#         normalized_tokens=["john", "smith"],
#         normalized_start_index=0,
#         normalized_end_index=10,
#     )
#
#     # Same tokens
#     span2 = Span(
#         entity_type="PERSON",
#         entity_value="John Smith",
#         start_position=0,
#         end_position=10,
#         normalized_tokens=["john", "smith"],
#         normalized_start_index=0,
#         normalized_end_index=10,
#     )
#
#     # Subset of tokens
#     span3 = Span(
#         entity_type="PERSON",
#         entity_value="John",
#         start_position=0,
#         end_position=4,
#         normalized_tokens=["john"],
#         normalized_start_index=0,
#         normalized_end_index=4,
#     )
#
#     # Different tokens
#     span4 = Span(
#         entity_type="PERSON",
#         entity_value="Mary",
#         start_position=15,
#         end_position=19,
#         normalized_tokens=["mary"],
#         normalized_start_index=15,
#         normalized_end_index=19,
#     )
#
#     # Test token-based IoU calculations
#     iou_exact = SpanEvaluator.calculate_iou(span1, span2, char_based=False)
#     iou_partial = SpanEvaluator.calculate_iou(span1, span3, char_based=False)
#     iou_none = SpanEvaluator.calculate_iou(span1, span4, char_based=False)
#
#     assert iou_exact > 0.5  # Should be high for exact match
#     assert 0 < iou_partial < iou_exact  # Should be lower for partial match
#     assert iou_none == 0.0  # Should be zero for no overlap
#
#
# @pytest.mark.parametrize(
#     "ann_spans, pred_spans, expected_pii_tp, expected_pii_annotated, expected_pii_predicted, expected_precision, expected_recall, expected_results, test_description",
#     [
#         # Original test case with specific entity types
#         (
#             [
#                 Span(
#                     entity_type="PERSON",
#                     entity_value="John Smith",
#                     start_position=0,
#                     end_position=10,
#                     normalized_tokens=["john", "smith"],
#                     normalized_start_index=0,
#                     normalized_end_index=10,
#                 ),
#                 Span(
#                     entity_type="LOCATION",
#                     entity_value="London",
#                     start_position=20,
#                     end_position=26,
#                     normalized_tokens=["london"],
#                     normalized_start_index=20,
#                     normalized_end_index=26,
#                 ),
#             ],
#             [
#                 Span(
#                     entity_type="PERSON",
#                     entity_value="John Smith",
#                     start_position=0,
#                     end_position=10,
#                     normalized_tokens=["john", "smith"],
#                     normalized_start_index=0,
#                     normalized_end_index=10,
#                 ),
#                 Span(
#                     entity_type="ORGANIZATION",
#                     entity_value="London",
#                     start_position=20,
#                     end_position=26,
#                     normalized_tokens=["london"],
#                     normalized_start_index=20,
#                     normalized_end_index=26,
#                 ),
#                 Span(
#                     entity_type="DATE",
#                     entity_value="2023",
#                     start_position=30,
#                     end_position=34,
#                     normalized_tokens=["2023"],
#                     normalized_start_index=30,
#                     normalized_end_index=34,
#                 ),
#             ],
#             2,  # Both PERSON and LOCATION are PII (even with type mismatch)
#             2,  # PERSON and LOCATION annotated
#             3,  # PERSON, ORGANIZATION, DATE predicted
#             2 / 3,  # 2 correct PII out of 3 predicted
#             1.0,  # All annotated PII detected
#             {
#                 ("PERSON", "PERSON"): 1,
#                 ("LOCATION", "ORGANIZATION"): 1,
#                 ("O", "DATE"): 1,
#             },
#             "Entity-specific matching with type mismatches",
#         ),
#         # New test case with global PII/O labels
#         (
#             [
#                 Span(
#                     entity_type="PII",
#                     entity_value="John Smith",
#                     start_position=0,
#                     end_position=10,
#                     normalized_tokens=["john", "smith"],
#                     normalized_start_index=0,
#                     normalized_end_index=10,
#                 ),
#                 Span(
#                     entity_type="PII",
#                     entity_value="London",
#                     start_position=20,
#                     end_position=26,
#                     normalized_tokens=["london"],
#                     normalized_start_index=20,
#                     normalized_end_index=26,
#                 ),
#             ],
#             [
#                 Span(
#                     entity_type="PII",
#                     entity_value="John Smith",
#                     start_position=0,
#                     end_position=10,
#                     normalized_tokens=["john", "smith"],
#                     normalized_start_index=0,
#                     normalized_end_index=10,
#                 ),
#                 Span(
#                     entity_type="O",
#                     entity_value="London",
#                     start_position=20,
#                     end_position=26,
#                     normalized_tokens=["london"],
#                     normalized_start_index=20,
#                     normalized_end_index=26,
#                 ),
#                 Span(
#                     entity_type="PII",
#                     entity_value="2023",
#                     start_position=30,
#                     end_position=34,
#                     normalized_tokens=["2023"],
#                     normalized_start_index=30,
#                     normalized_end_index=34,
#                 ),
#             ],
#             1,  # Only one PII correctly identified
#             2,  # Two PII spans annotated
#             2,  # Two PII spans predicted (one correct, one FP)
#             0.5,  # 1 correct out of 2 predicted
#             0.5,  # 1 detected out of 2 annotated
#             {("PII", "PII"): 1, ("PII", "O"): 1, ("O", "PII"): 1},
#             "Global PII/O label matching",
#         ),
#     ],
# )
# def test_match_predictions_with_annotations(
#     mock_span_evaluator,
#     ann_spans,
#     pred_spans,
#     expected_pii_tp,
#     expected_pii_annotated,
#     expected_pii_predicted,
#     expected_precision,
#     expected_recall,
#     expected_results,
#     test_description,
# ):
#     """Test the core matching logic between annotations and predictions."""
#
#     result = EvaluationResult()
#     result = mock_span_evaluator._match_predictions_with_annotations(
#         ann_spans, pred_spans, result
#     )
#
#     print(f"Testing: {test_description}")
#
#     # Check confusion matrix results
#     for (ann, pred), expected_count in expected_results.items():
#         actual_count = result.results.get((ann, pred), 0)
#         print(
#             f"{ann}-{pred} matches - Expected: {expected_count}, Actual: {actual_count}"
#         )
#         assert actual_count == expected_count
#
#     # Check global PII metrics
#     print(
#         f"PII True Positives - Expected: {expected_pii_tp}, Actual: {result.pii_true_positives}"
#     )
#     assert result.pii_true_positives == expected_pii_tp
#
#     print(
#         f"PII Annotated - Expected: {expected_pii_annotated}, Actual: {result.pii_annotated}"
#     )
#     assert result.pii_annotated == expected_pii_annotated
#
#     print(
#         f"PII Predicted - Expected: {expected_pii_predicted}, Actual: {result.pii_predicted}"
#     )
#     assert result.pii_predicted == expected_pii_predicted
#
#     print(
#         f"PII Precision - Expected: {expected_precision:.3f}, Actual: {result.pii_precision:.3f}"
#     )
#     assert result.pii_precision == pytest.approx(expected_precision, abs=1e-3)
#
#     print(
#         f"PII Recall - Expected: {expected_recall:.3f}, Actual: {result.pii_recall:.3f}"
#     )
#     assert result.pii_recall == pytest.approx(expected_recall, abs=1e-3)
#
#     # Check error analysis population
#     expected_error_count = (
#         len([count for count in expected_results.values() if count > 0]) - 1
#     )  # Subtract correct matches
#     print(
#         f"Model errors count - Expected: {expected_error_count}, Actual: {len(result.model_errors)}"
#     )
#     assert len(result.model_errors) == expected_error_count
#
#
# # ===== Integration Tests =====
#
#
# def test_calculate_score_on_df_with_perfect_match(mock_span_evaluator, sample_df):
#     """Test end-to-end evaluation with perfect prediction-annotation match."""
#     # Run evaluation
#     result = EvaluationResult()
#     result = mock_span_evaluator.calculate_score_on_df(
#         per_type=True, results_df=sample_df, evaluation_result=result
#     )
#     global_pii_df = mock_span_evaluator.create_global_entities_df(results_df=sample_df)
#     result = mock_span_evaluator.calculate_score_on_df(
#         per_type=False, results_df=global_pii_df, evaluation_result=result
#     )
#
#     # Check key metrics
#     assert result.pii_annotated == 2  # PERSON and LOCATION
#     assert result.pii_predicted == 2
#     assert result.pii_true_positives == 2
#     assert result.pii_precision == 1.0
#     assert result.pii_recall == 1.0
#     assert result.pii_f == 1.0
#
#     # Check per-type metrics
#     assert "PERSON" in result.per_type
#     assert "LOCATION" in result.per_type
#     assert result.per_type["PERSON"].precision == 1.0
#     assert result.per_type["PERSON"].recall == 1.0
#     assert result.per_type["LOCATION"].precision == 1.0
#     assert result.per_type["LOCATION"].recall == 1.0
#
#
# def test_calculate_score_on_df_with_errors():
#     """Test end-to-end evaluation with various error types."""
#     # Create a DataFrame with different types of errors
#     df = pd.DataFrame(
#         {
#             "sentence_id": [0] * 8,
#             "token": ["John", "Smith", "lives", "in", "New", "York", "City", "!"],
#             "annotation": [
#                 "PERSON",
#                 "PERSON",
#                 "O",
#                 "O",
#                 "LOCATION",
#                 "LOCATION",
#                 "LOCATION",
#                 "O",
#             ],
#             "prediction": [
#                 "PERSON",
#                 "O",
#                 "O",
#                 "O",
#                 "ORGANIZATION",
#                 "ORGANIZATION",
#                 "ORGANIZATION",
#                 "DATE",
#             ],
#             "start_indices": [0, 5, 11, 17, 20, 24, 29, 34],
#         }
#     )
#
#     evaluator = SpanEvaluator(model=MockModel(), iou_threshold=0.4, char_based=True)
#     result = EvaluationResult()
#     result = evaluator.calculate_score_on_df(
#         per_type=True, results_df=df, evaluation_result=result
#     )
#     global_pii_df = evaluator.create_global_entities_df(results_df=df)
#     result = evaluator.calculate_score_on_df(
#         per_type=False, results_df=global_pii_df, evaluation_result=result
#     )
#
#     # Check total counts
#     assert result.pii_annotated == 2  # PERSON and LOCATION spans
#     assert (
#         result.pii_predicted == 2
#     )  # PERSON, ORGANIZATION. ! is a skip word so DATE is ignored
#     assert (
#         result.pii_true_positives == 2
#     )  # Total true positives is regardless of type (PII yes/no)
#
#     # Check specific errors
#     assert result.results[("PERSON", "PERSON")] == 1
#     assert result.results[("LOCATION", "ORGANIZATION")] == 1  # Entity type mismatch
#     assert (
#         "O",
#         "DATE",
#     ) not in result.results  # DATE is a skip word, so it should not be counted
#
#     # Check error analysis
#     assert len(result.model_errors) == 1  # Location/Organization mismatch
#     # Check metrics
#     assert result.pii_precision == 1.0  # PII was fully predicted
#     assert result.pii_recall == 1.0  # Pii was fully covered
#
#
# def test_calculate_score_on_df_per_type_metrics_are_correct(
#     mock_span_evaluator, sample_df
# ):
#     """Test that per-type metrics are correctly calculated."""
#     result = EvaluationResult()
#     result = mock_span_evaluator.calculate_score_on_df(
#         per_type=True, results_df=sample_df, evaluation_result=result
#     )
#     global_pii_df = mock_span_evaluator.create_global_entities_df(results_df=sample_df)
#     result = mock_span_evaluator.calculate_score_on_df(
#         per_type=False, results_df=global_pii_df, evaluation_result=result
#     )
#
#     # Check per-type metrics
#     assert "PERSON" in result.per_type
#     assert "LOCATION" in result.per_type
#
#     person_metrics = result.per_type["PERSON"]
#     location_metrics = result.per_type["LOCATION"]
#
#     assert person_metrics.precision == 1.0
#     assert person_metrics.recall == 1.0
#     assert person_metrics.f_beta == 1.0
#
#     assert location_metrics.precision == 1.0
#     assert location_metrics.recall == 1.0
#     assert location_metrics.f_beta == 1.0
#
#
# def test_corner_case_empty_dataset():
#     """Test behavior when processing an empty dataset."""
#     df = pd.DataFrame(
#         {
#             "sentence_id": [],
#             "token": [],
#             "annotation": [],
#             "prediction": [],
#             "start_indices": [],
#         }
#     )
#
#     evaluator = SpanEvaluator(model=MockModel(), iou_threshold=0.5)
#     result = EvaluationResult()
#     result = evaluator.calculate_score_on_df(
#         per_type=True, results_df=df, evaluation_result=result
#     )
#     global_pii_df = evaluator.create_global_entities_df(results_df=df)
#     result = evaluator.calculate_score_on_df(
#         per_type=False, results_df=global_pii_df, evaluation_result=result
#     )
#
#     # Check that metrics are properly initialized for empty data
#     assert result.pii_annotated == 0
#     assert result.pii_predicted == 0
#     assert result.pii_true_positives == 0
#     assert np.isnan(result.pii_precision)
#     assert np.isnan(result.pii_recall)
#     assert isinstance(result.pii_f, float)  # Should be NaN or 0.0
#
#
# def test_corner_case_unicode_handling():
#     """Test handling of Unicode characters in text."""
#     df = pd.DataFrame(
#         {
#             "sentence_id": [0, 0, 0],
#             "token": ["José", "Martínez", "España"],
#             "annotation": ["PERSON", "PERSON", "LOCATION"],
#             "prediction": ["PERSON", "PERSON", "LOCATION"],
#             "start_indices": [0, 5, 14],
#         }
#     )
#
#     evaluator = SpanEvaluator(model=MockModel(), iou_threshold=0.5)
#     result = EvaluationResult()
#     result = evaluator.calculate_score_on_df(
#         per_type=True, results_df=df, evaluation_result=result
#     )
#     global_pii_df = evaluator.create_global_entities_df(results_df=df)
#     result = evaluator.calculate_score_on_df(
#         per_type=False, results_df=global_pii_df, evaluation_result=result
#     )
#
#     # Check Unicode is properly handled
#     assert result.pii_annotated == 2  # PERSON and LOCATION
#     assert result.pii_predicted == 2
#     assert result.pii_true_positives == 2
#     assert result.pii_precision == 1.0
#     assert result.pii_recall == 1.0
#
#
# def test_span_edge_cases():
#     """Test edge cases in span creation and matching."""
#     # Create a DataFrame with unusual token patterns
#     df = pd.DataFrame(
#         {
#             "sentence_id": [0] * 7,
#             "token": ["", "John", " ", "-", "", "Smith", ""],  # Empty tokens, spaces
#             "annotation": ["O", "PERSON", "O", "O", "O", "PERSON", "O"],
#             "prediction": ["O", "PERSON", "O", "O", "O", "PERSON", "O"],
#             "start_indices": [0, 0, 4, 5, 6, 6, 11],
#         }
#     )
#
#     evaluator = SpanEvaluator(model=MockModel(), iou_threshold=0.5)
#
#     # Test span creation with unusual tokens
#     spans = evaluator._create_spans(df, "annotation")
#     spans = evaluator._merge_adjacent_spans(spans, df)
#
#     # Check that empty tokens are handled properly
#     assert len(spans) == 1
#     assert spans[0].normalized_tokens == ["john", "smith"]
#
#     # Run full evaluation
#     result = EvaluationResult()
#     result = evaluator.calculate_score_on_df(
#         per_type=True, results_df=df, evaluation_result=result
#     )
#     global_pii_df = evaluator.create_global_entities_df(results_df=df)
#     result = evaluator.calculate_score_on_df(
#         per_type=False, results_df=global_pii_df, evaluation_result=result
#     )
#
#     assert result.pii_annotated == 1
#     assert result.pii_predicted == 1
#     assert result.pii_true_positives == 1
#
#
# # ===== Error Analysis Tests =====
#
#
# # fmt: off
# @pytest.mark.parametrize(
#     "annotation, "
#     "prediction, "
#     "tokens, "
#     "start_indices, "
#     "expected_errors_types, "
#     "expected_results, "
#     "expected_per_type",
#     [
#         # FALSE POSITIVE TEST CASE
#         (
#             ["PERSON", "PERSON", "O", "O", "O"],
#             ["PERSON", "PERSON", "O", "O", "LOCATION"],
#             ["John", "Smith", "went", "to", "school"],
#             [0, 5, 11, 16, 19],
#             [ErrorType.FP],
#             {
#                 ("PERSON", "PERSON"): 1,
#                 ("O", "LOCATION"): 1},
#             {
#                 "PERSON": {"tp": 1, "fp": 0, "fn": 0},
#                 "LOCATION": {"tp": 0, "fp": 1, "fn": 0},
#             },
#         ),
#         # FALSE NEGATIVE TEST CASE
#         (
#             ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
#             ["PERSON", "PERSON", "O", "O", "O"],
#             ["John", "Smith", "went", "to", "Paris"],
#             [0, 5, 11, 16, 19],
#             [ErrorType.FN],
#             {
#                 ("PERSON", "PERSON"): 1,
#                 ("LOCATION", "O"): 1
#             },
#             {
#                 "PERSON": {"tp": 1, "fp": 0, "fn": 0},
#                 "LOCATION": {"tp": 0, "fp": 0, "fn": 1},
#             },
#         ),
#         # WRONG ENTITY TYPE TEST CASE
#         (
#             ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
#             ["PERSON", "PERSON", "O", "ORGANIZATION", "ORGANIZATION"],
#             ["John", "Smith", "works", "at", "Microsoft"],
#             [0, 5, 11, 17, 20],
#             [
#                 ErrorType.WrongEntity
#             ],  # Only check error types, not the full error objects
#             {
#                 ("PERSON", "PERSON"): 1,
#                 ("LOCATION", "ORGANIZATION"): 1
#             },
#             {
#                 "PERSON": {"tp": 1, "fp": 0, "fn": 0},
#                 "LOCATION": {"tp": 0, "fp": 0, "fn": 1},
#                 "ORGANIZATION": {"tp": 0, "fp": 1, "fn": 0},
#             },
#         ),
#         # MULTIPLE ERROR TYPES COMBINED
#         (
#             ["PERSON", "PERSON", "O", "DATE", "DATE", "O", "LOCATION", "O"],
#             ["PERSON", "PERSON", "O", "PHONE", "PHONE", "EMAIL", "O", "LOCATION"],
#             ["John", "Smith", "born", "May", "1980", "visited", "Paris", "recently"],
#             [0, 5, 11, 16, 20, 25, 33, 39],
#             [
#                 ErrorType.WrongEntity,
#                 ErrorType.FP,
#                 ErrorType.FN,
#             ],
#             {
#                 ("PERSON", "PERSON"): 1,
#                 ("DATE", "PHONE"): 1,
#                 ("O", "EMAIL"): 1,
#                 ("LOCATION", "O"): 1,
#                 ("O", "LOCATION"): 1,
#             },
#             {
#                 "PERSON": {"tp": 1, "fp": 0, "fn": 0},
#                 "DATE": {"tp": 0, "fp": 0, "fn": 1},
#                 "PHONE": {"tp": 0, "fp": 1, "fn": 0},
#                 "EMAIL": {"tp": 0, "fp": 1, "fn": 0},
#                 "LOCATION": {"tp": 0, "fp": 1, "fn": 1},
#             },
#         ),
#         # LOW IoU cases - special case that behaves differently
#         # The implementation handles this case specially - we only get PERSON match
#         (
#             ["PERSON", "PERSON", "PERSON", "O"],
#             ["O", "PERSON", "PERSON", "O"],
#             ["John", "Robert", "Smith", "here"],
#             [0, 5, 12, 18],
#             [],  # No errors in this case - IoU is sufficient for match
#             None,  # Skip exact counts check - implementation behavior may change
#             {
#                 "PERSON": {"tp": 1, "fp": 0, "fn": 0}  # IoU threshold allows the match
#             },
#         ),
#         # LOW IoU FALSE POSITIVE TEST CASE - simplified
#         (
#             ["O", "PERSON", "PERSON", "O"],
#             ["PERSON", "PERSON", "O", "O"],
#             ["Mr.", "John", "Smith", "here"],
#             [0, 4, 9, 15],
#             [ErrorType.FN, ErrorType.FP],
#             {
#                 ("O", "PERSON"): 1,
#                 ("PERSON", "O"): 1,
#             },
#             {
#                 "PERSON": {"tp": 0, "fp": 1, "fn": 1},
#             },
#         ),
#         # COMBINED LOW IoU CASE - Both FP and FN due to boundary mismatches - simplified
#         (
#             ["DATE", "DATE", "DATE", "O", "O", "PERSON", "PERSON", "O"],
#             ["O", "DATE", "DATE", "DATE", "O", "O", "PERSON", "PERSON"],
#             ["January", "15", "2023", "is", "when", "John", "Smith", "arrived"],
#             [0, 8, 11, 16, 19, 24, 29, 35],
#             [ErrorType.FN, ErrorType.FP],
#             {
#                 ("DATE", "O"): 1,
#                 ("O", "DATE"): 1,
#                 ("PERSON", "O"): 1,
#                 ("O", "PERSON"): 1,
#             },
#             {
#                 "DATE": {"tp": 0, "fp": 1, "fn": 1},
#                 "PERSON": {"tp": 0, "fp": 1, "fn": 1},
#             },
#         ),
#     ],
# )
# def test_error_analysis(
#     mock_span_evaluator,
#     annotation,
#     prediction,
#     tokens,
#     start_indices,
#     expected_errors_types,
#     expected_results,
#     expected_per_type,
# ):
#     """
#     Test that error analysis correctly identifies and records all three types of errors:
#
#     1. False Positives (FP): Predicted entities that don't exist in annotations
#     2. False Negatives (FN): Annotated entities that weren't predicted
#     3. Wrong Entity Type (WrongEntity): Entity boundaries match but types differ
#
#     Focuses specifically on verifying:
#     - Error analysis records contain expected error types
#     - Confusion matrix (results dictionary) when applicable
#     - Per-entity type metric population when applicable
#     """
#     # Create the test dataframe
#     df = pd.DataFrame(
#         {
#             "sentence_id": [0] * len(tokens),
#             "token": tokens,
#             "annotation": annotation,
#             "prediction": prediction,
#             "start_indices": start_indices,
#         }
#     )
#
#     # Run evaluation
#     eval_result = EvaluationResult()
#     eval_result = mock_span_evaluator.calculate_score_on_df(per_type=True, results_df=df, evaluation_result=eval_result)
#     global_pii_df = mock_span_evaluator.create_global_entities_df(results_df=df)
#     eval_result = mock_span_evaluator.calculate_score_on_df(per_type=False, results_df=global_pii_df, evaluation_result=eval_result)
#     # Verify expected error types are present (not the exact count)
#     error_types = [error.error_type for error in eval_result.model_errors]
#     for expected_type in expected_errors_types:
#         assert (
#             expected_type in error_types
#         ), f"Expected error type {expected_type} not found in results"
#
#     # Verify confusion matrix entries (if specified)
#     if expected_results:
#         for (ann, pred), count in expected_results.items():
#             assert (
#                 eval_result.results.get((ann, pred), 0) == count
#             ), f"Expected {count} for ({ann}, {pred}), got {eval_result.results.get((ann, pred), 0)}"
#
#     # Verify per-type metrics (if specified)
#     if expected_per_type:
#         for entity_type, metrics in expected_per_type.items():
#             assert (
#                 entity_type in eval_result.per_type
#             ), f"Entity type {entity_type} not found in results"
#
#             if "tp" in metrics:
#                 assert (
#                     eval_result.per_type[entity_type].true_positives == metrics["tp"]
#                 ), (
#                     f"Expected {metrics['tp']} true positives for {entity_type}, "
#                     f"got {eval_result.per_type[entity_type].true_positives}"
#                 )
#
#             if "fp" in metrics:
#                 assert (
#                     eval_result.per_type[entity_type].false_positives == metrics["fp"]
#                 ), (
#                     f"Expected {metrics['fp']} false positives for {entity_type}, "
#                     f"got {eval_result.per_type[entity_type].false_positives}"
#                 )
#
#             if "fn" in metrics:
#                 assert (
#                     eval_result.per_type[entity_type].false_negatives == metrics["fn"]
#                 ), (
#                     f"Expected {metrics['fn']} false negatives for {entity_type}, "
#                     f"got {eval_result.per_type[entity_type].false_negatives}"
#                 )
#
#
#
# def test_mixture_of_entities_global_metrics_are_high():
#     """Test that the evaluator can handle a mixture of entity types."""
#     df = pd.DataFrame(
#         {
#             "sentence_id": [0]*19,
#             "token": [     "The","address", "of",    "Persint",      "is",  "6750",     "Koskikatu",    "25",	    "Apt",      ".",	    "864",      "",         "Artilleros","",        ",",        "CO",       "",	        "Uruguay",  "64677"], # noqa: E501
#             "annotation": ["O" , "O",	    "O",	 "ORGANIZATION", "O",	"LOCATION",	"LOCATION",	    "LOCATION",	"LOCATION",	"LOCATION",	"LOCATION",	"LOCATION",	"LOCATION",	"LOCATION",	"LOCATION",	"LOCATION", "LOCATION",	"LOCATION",	"LOCATION"],  # noqa: E501
#             "prediction": ["O" , "O",	    "O",	 "PERSON",    	 "O",	"LOCATION",	"LOCATION",	    "O",	    "LOCATION",	"LOCATION",	"LOCATION",	"LOCATION",	"O",        "LOCATION",	"LOCATION", "LOCATION",	"LOCATION",	"LOCATION",	"PHONE_NUMBER",], # noqa: E501
#             "start_indices": [0,  4,	    12,	     15,	         23,	26,	        31,	            41,	        44,	        47,	        49,	        52,	        53,	        63,	        64,	        66,	        68,	        70,	        78], # noqa: E501
#         }
#     )
#     evaluator = SpanEvaluator(model=MockModel(), iou_threshold=0.5)
#     result = evaluator.calculate_score_on_df(per_type=True, results_df=df)
#     global_pii_df = evaluator.create_global_entities_df(results_df=df)
#     result = evaluator.calculate_score_on_df(
#         per_type=False,
#         results_df=global_pii_df,
#         evaluation_result=result
#     )
#     assert result.per_type["LOCATION"].f_beta > 0.5
#     assert result.pii_precision > 0.8
#     assert result.pii_recall > 0.8
# # fmt: on
