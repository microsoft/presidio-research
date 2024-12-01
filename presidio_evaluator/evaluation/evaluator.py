from collections import Counter
from typing import List, Optional, Dict, Union

import numpy as np
from presidio_analyzer import AnalyzerEngine

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import EvaluationResult, ModelError, ErrorType
from presidio_evaluator.evaluation.skipwords import get_skip_words
from presidio_evaluator.models import BaseModel, PresidioAnalyzerWrapper

GENERIC_ENTITIES = ("PII", "ID", "PII", "PHI", "ID_NUM", "NUMBER", "NUM", "GENERIC_PII")


class Evaluator:
    def __init__(
        self,
        model: Union[BaseModel, AnalyzerEngine],
        verbose: bool = False,
        compare_by_io=True,
        entities_to_keep: Optional[List[str]] = None,
        generic_entities: Optional[List[str]] = None,
        skip_words: Optional[List] = None,
    ):
        """
        Evaluate a PII detection model or a Presidio analyzer / recognizer

        :param model: Instance of a fitted model (of base type BaseModel),
        or an instance of Presidio Analyzer
        :param compare_by_io: True if comparison should be done on the entity
        level and not the sub-entity level
        :param entities_to_keep: List of entity names to focus the evaluator on (and ignore the rest).
        Default is None = all entities. If the provided model has a list of entities to keep,
        this list would be used for evaluation.
        :param generic_entities: List of entities that are not considered an error if
        detected instead of something other entity. For example: PII, ID, number
        :param skip_words: List of words to skip. If None, the default list would be used.
        """
        if isinstance(model, AnalyzerEngine):
            self.model = PresidioAnalyzerWrapper(analyzer_engine=model)
        elif isinstance(model, BaseModel):
            self.model = model
        else:
            raise ValueError(
                "Model should be an instance of BaseModel or Presidio Analyzer"
            )

        self.verbose = verbose
        self.compare_by_io = compare_by_io
        self.entities_to_keep = entities_to_keep
        if self.entities_to_keep is None and self.model.entities:
            self.entities_to_keep = self.model.entities

        self.generic_entities = (
            generic_entities if generic_entities else GENERIC_ENTITIES
        )

        self.skip_words = skip_words if skip_words else get_skip_words()

    def compare(self, input_sample: InputSample, prediction: List[str]):
        """
        Compares ground truth tags (annotation) and predicted (prediction)
        :param input_sample: input sample containing list of tags with scheme
        :param prediction: predicted value for each token
        self.labeling_scheme

        """
        annotation = input_sample.tags
        tokens = input_sample.tokens

        if len(annotation) != len(prediction):
            print(
                "Annotation and prediction do not have the"
                "same length. Sample={}".format(input_sample)
            )
            return Counter(), []

        results = Counter()
        mistakes = []

        new_annotation = annotation.copy()

        if self.compare_by_io:
            new_annotation = self._to_io(new_annotation)
            prediction = self._to_io(prediction)

        # Ignore annotations that aren't in the list of
        # requested entities.
        if self.entities_to_keep:
            prediction = self._adjust_per_entities(prediction)
            new_annotation = self._adjust_per_entities(new_annotation)

        for i in range(0, len(new_annotation)):
            cur_token = tokens[i]
            cur_prediction = prediction[i]
            cur_annotation = new_annotation[i]
            results[(cur_annotation, cur_prediction)] += 1

            if self.verbose:
                print("Annotation:", cur_annotation)
                print("Prediction:", cur_prediction)
                print(results)

            # check if there was an error
            is_error = cur_annotation != cur_prediction

            if is_error:
                reverted = self.__revert_known_errors(
                    cur_annotation, cur_prediction, cur_token, results
                )
                if reverted:
                    # This isn't really an error, continue.
                    continue

                if prediction[i] == "O":
                    mistakes.append(
                        ModelError(
                            error_type=ErrorType.FN,
                            annotation=cur_annotation,
                            prediction=cur_prediction,
                            token=cur_token,
                            full_text=input_sample.full_text,
                            metadata=input_sample.metadata,
                        )
                    )
                elif new_annotation[i] == "O":
                    mistakes.append(
                        ModelError(
                            error_type=ErrorType.FP,
                            annotation=cur_annotation,
                            prediction=cur_prediction,
                            token=cur_token,
                            full_text=input_sample.full_text,
                            metadata=input_sample.metadata,
                        )
                    )
                else:
                    mistakes.append(
                        ModelError(
                            error_type=ErrorType.WrongEntity,
                            annotation=cur_annotation,
                            prediction=cur_prediction,
                            token=cur_token,
                            full_text=input_sample.full_text,
                            metadata=input_sample.metadata,
                        )
                    )

        return results, mistakes

    def __revert_known_errors(
        self, current_annotation, current_prediction, current_token, results
    ) -> bool:
        reverted = False

        if str(current_token).lower().strip() in self.skip_words:
            # Ignore cases where the token is a skip word
            results[(current_annotation, current_prediction)] -= 1
            reverted = True

        if current_prediction in self.generic_entities and current_annotation != "O":
            # Ignore cases where the prediction is generic
            results[(current_annotation, current_prediction)] -= 1
            # Add a result which assumes the generic equals the specific
            results[(current_annotation, current_annotation)] += 1
            reverted = True

        elif current_annotation in self.generic_entities and current_prediction != "O":
            # Ignore cases where the prediction is generic
            results[(current_annotation, current_prediction)] -= 1
            # Add a result which assumes the generic equals the specific
            results[(current_prediction, current_prediction)] += 1
            reverted = True

        # Remove temporary keys which should not be counted
        if results[(current_annotation, current_prediction)] == 0:
            del results[(current_annotation, current_prediction)]

        return reverted

    def _adjust_per_entities(self, tags):
        if self.entities_to_keep:
            return [tag if tag in self.entities_to_keep else "O" for tag in tags]
        else:
            return tags

    @staticmethod
    def _to_io(tags):
        """
        Translates BILUO/BIO/IOB to IO - only In or Out of entity.
        ['B-PERSON','I-PERSON','L-PERSON'] is translated into
        ['PERSON','PERSON','PERSON']
        :param tags: the input tags in BILUO/IOB/BIO format
        :return: a new list of IO tags
        """
        return [tag[2:] if "-" in tag else tag for tag in tags]

    def evaluate_sample(
        self, sample: InputSample, prediction: List[str]
    ) -> EvaluationResult:
        if self.verbose:
            print("Input sentence: {}".format(sample.full_text))

        results, mistakes = self.compare(input_sample=sample, prediction=prediction)
        return EvaluationResult(results, mistakes, sample.full_text)

    def evaluate_all(
        self, dataset: List[InputSample], **kwargs
    ) -> List[EvaluationResult]:
        """Evaluate a dataset given a model and labels.

        :param dataset: A list of InputSample samples, containing the ground truth tags
        :param kwargs: Additional arguments for the model's predict method
        """

        evaluation_results = []
        if self.model.entity_mapping:
            print(
                f"Mapping entity values using this dictionary: {self.model.entity_mapping}"
            )

        print(f"Running model {self.model.__class__.__name__} on dataset...")
        predictions = self.model.batch_predict(dataset, **kwargs)
        print("Finished running model on dataset")

        for prediction, sample in zip(predictions, dataset):

            # Remove entities not requested (in model.entities_to_keep))
            prediction = self.model.filter_tags_in_supported_entities(prediction)

            # Switch to requested labeling scheme (IO/BIO/BILUO)
            prediction = self.model.to_scheme(prediction)

            evaluation_result = self.evaluate_sample(
                sample=sample, prediction=prediction
            )
            evaluation_results.append(evaluation_result)

        return evaluation_results

    @staticmethod
    def align_entity_types(
        input_samples: List[InputSample],
        entities_mapping: Dict[str, str] = None,
        allow_missing_mappings: bool = False,
    ) -> List[InputSample]:
        """
        Change input samples to conform with Presidio's entities
        :return: new list of InputSample
        """

        new_input_samples = input_samples.copy()

        def is_key_in_dict(key_dict: Dict[str, str], search_key: str):
            """Check if a key is in a dictionary, ignoring case and underscores."""
            # Normalize the search key by converting to uppercase and removing underscores
            normalized_search_key = search_key.upper().replace("_", "")

            # Check if any normalized key in the dictionary matches the search key
            return any(
                normalized_search_key == key.upper().replace("_", "")
                for key in key_dict.keys()
            )

        # A list that will contain updated input samples,
        new_list = []

        for input_sample in new_input_samples:
            contains_field_in_mapping = False
            new_spans = []
            # Update spans to match the entity types in the values of entities_mapping
            for span in input_sample.spans:
                if is_key_in_dict(entities_mapping, span.entity_type):
                    new_name = entities_mapping.get(span.entity_type)
                    span.entity_type = new_name
                    contains_field_in_mapping = True

                    new_spans.append(span)
                else:
                    if not allow_missing_mappings:
                        raise ValueError(
                            f"Key {span.entity_type} cannot be found in the provided entities_mapping"
                        )
            input_sample.spans = new_spans

            # Update tags in case this sample has relevant entities for evaluation
            if contains_field_in_mapping:
                for i, tag in enumerate(input_sample.tags):
                    has_prefix = "-" in tag
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
                        input_sample.tags[i] = "O"

            new_list.append(input_sample)

        return new_list
        # Iterate on all samples

    def calculate_score(
        self,
        evaluation_results: List[EvaluationResult],
        entities: Optional[List[str]] = None,
        beta: float = 2.0,
    ) -> EvaluationResult:
        """
        Returns the pii_precision, pii_recall, f_measure either and number of records for each entity
        or for all entities (ignore_entity_type = True)
        :param evaluation_results: List of EvaluationResult
        :param entities: List of entities to calculate score to. Default is None: all entities
        :param beta: F measure beta value
        between different entity types, or to treat these as misclassifications
        :return: EvaluationResult with precision, recall and f measures
        """

        # aggregate results
        all_results = sum([er.results for er in evaluation_results], Counter())

        # compute pii_recall per entity
        entity_recall = {}
        entity_precision = {}
        n = {}
        if not entities:
            entities1 = list(set([x[0] for x in all_results.keys() if x[0] != "O"]))
            entities2 = list(set([x[1] for x in all_results.keys() if x[1] != "O"]))
            entities = list(set(entities1).union(set(entities2)))

        for entity in entities:
            # all annotation of given type
            annotated = sum([all_results[x] for x in all_results if x[0] == entity])
            predicted = sum([all_results[x] for x in all_results if x[1] == entity])
            n[entity] = annotated
            tp = all_results[(entity, entity)]

            if annotated > 0:
                entity_recall[entity] = tp / annotated
            else:
                entity_recall[entity] = np.nan

            if predicted > 0:
                per_entity_tp = all_results[(entity, entity)]
                entity_precision[entity] = per_entity_tp / predicted
            else:
                entity_precision[entity] = np.nan

        # compute pii_precision and pii_recall
        annotated_all = sum([all_results[x] for x in all_results if x[0] != "O"])
        predicted_all = sum([all_results[x] for x in all_results if x[1] != "O"])
        if annotated_all > 0:
            pii_recall = (
                sum(
                    [
                        all_results[x]
                        for x in all_results
                        if (x[0] != "O" and x[1] != "O")
                    ]
                )
                / annotated_all
            )
        else:
            pii_recall = np.nan
        if predicted_all > 0:
            pii_precision = (
                sum(
                    [
                        all_results[x]
                        for x in all_results
                        if (x[0] != "O" and x[1] != "O")
                    ]
                )
                / predicted_all
            )
        else:
            pii_precision = np.nan
        # compute pii_f_beta-score
        pii_f_beta = self.f_beta(pii_precision, pii_recall, beta)

        # aggregate errors
        errors = []
        for res in evaluation_results:
            if res.model_errors:
                errors.extend(res.model_errors)

        evaluation_result = EvaluationResult(
            results=all_results,
            model_errors=errors,
            pii_precision=pii_precision,
            pii_recall=pii_recall,
            entity_recall_dict=entity_recall,
            entity_precision_dict=entity_precision,
            n_dict=n,
            pii_f=pii_f_beta,
            n=sum(n.values()),
        )

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
        if np.isnan(precision) or np.isnan(recall) or (precision == 0 and recall == 0):
            return np.nan

        return ((1 + beta**2) * precision * recall) / (((beta**2) * precision) + recall)
