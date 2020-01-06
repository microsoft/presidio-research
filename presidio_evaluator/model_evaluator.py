from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd
from presidio_evaluator import InputSample, EvaluationResult, ModelError
from tqdm import tqdm


class ModelEvaluator(ABC):

    def __init__(self, entities_to_keep: List[str] = None,
                 verbose: bool = False,
                 use_spans: bool = False, labeling_scheme="BIO",
                 compare_by_io=True):

        """
        Abstract class for evaluating NER models and others
        :param entities_to_keep: Which entities should be evaluated? All other
        entities are ignored. If None, none are filtered
        :param verbose: Whether to print more debug info
        :param labeling_scheme: Type of scheme used for labeling (BILOU,
        BIO/LOB or IO)
        :param compare_by_io: True if comparison should be done on the entity
        level and not the sub-entity level

        """
        self.entities = entities_to_keep
        self.verbose = verbose
        self.use_spans = use_spans
        self.compare_by_io = compare_by_io
        self.labeling_scheme = labeling_scheme

    @abstractmethod
    def predict(self, sample: InputSample) -> List[str]:
        """
        Abstract. Returns the predicted tokens/spans from the evaluated model
        :param sample: Sample to be evaluated
        :return: if self.use spans: list of spans
                 if not self.use_spans: tags in self.labeling_scheme format
        """
        pass

    def compare(self, input_sample: InputSample, prediction: List[str]):

        """
        Compares gound truth tags (annotation) and predicted (prediction)
        :param input_sample: input sample containing list of tags with scheme
        :param prediction: predicted value for each token
        self.labeling_scheme

        """
        annotation = input_sample.tags
        tokens = input_sample.tokens

        if len(annotation) != len(prediction):
            print("Annotation and prediction do not have the"
                  "same length. Sample={}".format(input_sample))
            return Counter(), []

        results = Counter()
        mistakes = []

        new_annotation = annotation.copy()

        if self.compare_by_io:
            new_annotation = self._to_io(new_annotation)
            prediction = self._to_io(prediction)

        # Ignore annotations that aren't in the list of
        # requested entities.
        if self.entities:
            prediction = self._adjust_per_entities(prediction)
            new_annotation = self._adjust_per_entities(new_annotation)
        for i in range(0, len(new_annotation)):
            results[(new_annotation[i], prediction[i])] += 1

            if self.verbose:
                print('Annotation:', new_annotation[i])
                print('Prediction:', prediction[i])
                print(results)

            # check if there was an error
            is_error = (new_annotation[i] != prediction[i])
            if is_error:
                if prediction[i] == 'O':
                    mistakes.append(ModelError("FN",
                                               new_annotation[i],
                                               prediction[i],
                                               tokens[i],
                                               input_sample.full_text,
                                               input_sample.metadata))
                elif new_annotation[i] == 'O':
                    mistakes.append(ModelError("FP",
                                               new_annotation[i],
                                               prediction[i],
                                               tokens[i],
                                               input_sample.full_text,
                                               input_sample.metadata))
                else:
                    mistakes.append(ModelError("Wrong entity",
                                               new_annotation[i],
                                               prediction[i],
                                               tokens[i],
                                               input_sample.full_text,
                                               input_sample.metadata))

        return results, mistakes

    def _adjust_per_entities(self, tags):
        if self.entities:
            return [tag if tag in self.entities else 'O' for tag in tags]

    @staticmethod
    def _to_io(tags):
        """
        Translates BILOU/BIO/IOB to IO - only In or Out of entity.
        ['B-PERSON','I-PERSON','L-PERSON'] is translated into
        ['PERSON','PERSON','PERSON']
        :param tags: the input tags in BILOU/IOB/BIO format
        :return: a new list of IO tags
        """
        return [tag[2:] if '-' in tag else tag for tag in tags]

    def evaluate_sample(self, sample: InputSample) -> EvaluationResult:
        if self.verbose:
            print("Input sentence: {}".format(sample.full_text))

        prediction = self.predict(sample)
        results, mistakes = self.compare(
            input_sample=sample,
            prediction=prediction)
        return EvaluationResult(results, mistakes, sample.full_text)

    def evaluate_all(self, dataset: List[InputSample]) -> List[EvaluationResult]:
        evaluation_results = []
        for sample in tqdm(dataset, desc='Evaluating {}'.format(self.__class__)):
            evaluation_result = self.evaluate_sample(sample)
            evaluation_results.append(evaluation_result)

        return evaluation_results

    def calculate_score(self, evaluation_results: List[
        EvaluationResult], beta: float = 1) \
            -> EvaluationResult:
        """
        Returns the pii_precision, pii_recall and f_measure either for each entity
        or for all entities (ignore_entity_type = True)
        :param evaluation_results: List of EvaluationResult
        :param beta: F measure beta value
        between different entity types, or to treat these as misclassifications
        :return: EvaluationResult with precision, recall and f measures
        """

        # aggregate results
        all_results = sum([er.results for er in evaluation_results], Counter())

        # compute pii_recall per entity
        entity_recall = {}
        entity_precision = {}
        if self.entities:
            entities = self.entities
        else:
            entities = list(
                set([x[0] for x in all_results.keys() if x[0] != 'O']))

        for entity in entities:
            # all annotation of given type
            annotated = sum(
                [all_results[x] for x in all_results if x[0] == entity])
            predicted = sum(
                [all_results[x] for x in all_results if x[1] == entity])
            tp = all_results[(entity, entity)]

            if annotated > 0:
                entity_recall[entity] = tp / annotated
            else:
                entity_recall[entity] = np.NaN

            if predicted > 0:
                per_entity_tp = all_results[(entity, entity)]
                entity_precision[entity] = per_entity_tp / predicted
            else:
                entity_precision[entity] = np.NaN

        # compute pii_precision and pii_recall
        annotated_all = sum(
            [all_results[x] for x in all_results if x[0] != 'O'])
        predicted_all = sum(
            [all_results[x] for x in all_results if x[1] != 'O'])
        if annotated_all > 0:
            pii_recall = sum([all_results[x] for x in all_results if
                              (x[0] != 'O' and x[1] != 'O')]) / annotated_all
        else:
            pii_recall = np.NaN
        if predicted_all > 0:
            pii_precision = sum([all_results[x] for x in all_results if
                                 (x[0] != 'O' and x[1] != 'O')]) / predicted_all
        else:
            pii_precision = np.NaN
        # compute pii_f_beta-score
        pii_f_beta = self.f_beta(pii_precision, pii_recall, beta)

        # aggregate errors
        errors = []
        for res in evaluation_results:
            if res.model_errors:
                errors.extend(res.model_errors)

        evaluation_result = EvaluationResult(results=all_results, model_errors=errors)
        evaluation_result.pii_precision = pii_precision
        evaluation_result.pii_recall = pii_recall
        evaluation_result.entity_recall_dict = entity_recall
        evaluation_result.entity_precision_dict = entity_precision
        evaluation_result.pii_f = pii_f_beta

        return evaluation_result

    @staticmethod
    def precision(tp: int, fp: int) -> float:
        return tp / (tp + fp + 1e-100)

    @staticmethod
    def recall(tp: int, fn: int) -> float:
        return tp / (tp + fn + 1e-100)

    @staticmethod
    def f_beta(precision: float, recall: float, beta: float) -> float:
        """
        Returns the F score for precision, recall and a beta parameter
        :param precision: a float with the precision value
        :param recall: a float with the recall value
        :param beta: a float with the beta parameter of the F measure,
        which gives more or less weight to precision
        vs. recall
        :return: a float value of the f(beta) measure.
        """
        if np.isnan(precision) or np.isnan(recall) or (
                precision == 0 and recall == 0):
            return np.nan

        return ((1 + beta ** 2) * precision * recall) / (
                ((beta ** 2) * precision) + recall)

    @staticmethod
    def align_input_samples_to_presidio_analyzer(input_samples: List[InputSample],
                                                 entities_mapping: Dict[str, str],
                                                 presidio_fields: List[str]=None) \
            -> List[InputSample]:
        """
        Change input samples to conform with Presidio's entities
        :return: new list of InputSample
        """

        new_input_samples = input_samples.copy()

        # Match entity names to Presidio's
        if not presidio_fields:
            presidio_fields = ['CREDIT_CARD', 'CRYPTO', 'DATE_TIME', 'DOMAIN_NAME', 'EMAIL_ADDRESS', 'IBAN_CODE',
                           'IP_ADDRESS', 'NRP', 'LOCATION', 'PERSON', 'PHONE_NUMBER', 'US_SSN']

        # A list that will contain updated input samples,
        new_list = []

        # Iterate on all samples
        for input_sample in new_input_samples:
            contains_presidio_field = False
            new_spans = []
            # Update spans to match Presidio's entity name
            for span in input_sample.spans:
                in_presidio_field = False
                if span.entity_type in entities_mapping.keys():
                    new_name = entities_mapping.get(span.entity_type)
                    span.entity_type = new_name
                    contains_presidio_field = True

                    # Add to new span list, if the span contains an entity relevant to Presidio
                    new_spans.append(span)
            input_sample.spans = new_spans

            # Update tags in case this sample has relevant entities for evaluation
            if contains_presidio_field:
                for i, tag in enumerate(input_sample.tags):
                    has_prefix = '-' in tag
                    if has_prefix:
                        prefix = tag[:2]
                        clean = tag[2:]
                    else:
                        prefix = ""
                        clean = tag

                    if clean in entities_mapping.keys():
                        new_name = entities_mapping.get(clean)
                        input_sample.tags[i] = "{}{}".format(prefix, new_name)
                    else:
                        input_sample.tags[i] = 'O'

            new_list.append(input_sample)
        return new_list

    @staticmethod
    def get_false_positives(errors=List[ModelError], entity=None):
        """
        Get a list of all false positive errors in the results
        """
        if isinstance(entity, str):
            entity = [entity]

        if entity:
            return [model_error for model_error in errors if
                    model_error.error_type == 'FP' and model_error.prediction in entity]
        else:
            return [model_error for model_error in errors if model_error.error_type == 'FP']

    @staticmethod
    def get_false_negatives(errors=List[ModelError], entity=None):
        """
        Get a list of all false positive negative errors in the results (False negatives and wrong entity detection)
        """
        if isinstance(entity, str):
            entity = [entity]
        if entity:
            return [model_error for model_error in errors if
                    model_error.error_type != 'FP' and model_error.annotation in entity]
        else:
            return [model_error for model_error in errors if model_error.error_type != 'FP']

    @staticmethod
    def most_common_fp_tokens(errors=List[ModelError], n: int = 10, entity=None):
        """
        Print the n most common false positive tokens (tokens thought to be an entity)
        """
        fps = ModelEvaluator.get_false_positives(errors, entity)

        tokens = [err.token.text for err in fps]
        from collections import Counter
        by_frequency = Counter(tokens)
        most_common = by_frequency.most_common(n)
        print("Most common false positive tokens:")
        print(most_common)
        print("Example sentence with each FP token:")
        for tok, val in most_common:
            with_tok = [err for err in fps if err.token.text == tok]
            print(with_tok[0].full_text)

    @staticmethod
    def most_common_fn_tokens(errors=List[ModelError], n: int = 10, entity=None):
        """
        Print all tokens that were missed by the model, including an example of the full text in which they appear
        """
        fns = ModelEvaluator.get_false_negatives(errors, entity)

        fns_tokens = [err.token.text for err in fns]
        from collections import Counter
        by_frequency_fns = Counter(fns_tokens)
        most_common_fns = by_frequency_fns.most_common(50)
        print(most_common_fns)
        for tok, val in most_common_fns:
            with_tok = [err for err in fns if err.token.text == tok]
            print("Token: {}, Annotation: {}, Full text: {}".format(with_tok[0].token, with_tok[0].annotation,
                                                                    with_tok[0].full_text))

    @staticmethod
    def get_errors_df(errors=List[ModelError], entity: List[str] = None, error_type: str = 'FN'):
        """
        Get ModelErrors as pd.DataFrame
        """
        if error_type == 'FN':
            filtered_errors = ModelEvaluator.get_false_negatives(errors, entity)
        elif error_type == 'FP':
            filtered_errors = ModelEvaluator.get_false_positives(errors, entity)
        else:
            raise ValueError("error_type should be either FP or FN")

        if len(filtered_errors) == 0:
            print("No errors of type {} and entity {} were found".format(error_type,entity))
            return None

        errors_df = pd.DataFrame.from_records([error.__dict__ for error in filtered_errors])
        metadata_df = pd.DataFrame(errors_df['metadata'].tolist())
        errors_df.drop(['metadata'], axis=1, inplace=True)
        new_errors_df = pd.concat([errors_df, metadata_df], axis=1)
        return new_errors_df

    @staticmethod
    def get_fps_dataframe(errors=List[ModelError], entity: List[str] = None):
        """
        Get false positive ModelErrors as pd.DataFrame
        """
        return ModelEvaluator.get_errors_df(errors, entity, error_type='FP')

    @staticmethod
    def get_fns_dataframe(errors=List[ModelError], entity: List[str] = None):
        """
        Get false negative ModelErrors as pd.DataFrame
        """
        return ModelEvaluator.get_errors_df(errors, entity, error_type='FN')
