from typing import List

from presidio_analyzer import AnalyzerEngine

from presidio_evaluator import ModelEvaluator, InputSample, span_to_tag
from presidio_evaluator.data_generator import read_synth_dataset


class PresidioAnalyzerEvaluator(ModelEvaluator):
    def __init__(
        self,
        analyzer=AnalyzerEngine(),
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        labeling_scheme="BIO",
        compare_by_io=True,
        score_threshold=0.4,
    ):
        """
        Evaluation wrapper for the Presidio Analyzer
        :param analyzer: object of type AnalyzerEngine (from presidio-analyzer)
        """
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
            compare_by_io=compare_by_io,
        )
        self.analyzer = analyzer

        self.score_threshold = score_threshold

    def predict(self, sample: InputSample) -> List[str]:
        if self.entities is None or len(self.entities) == 0:
            all_fields = True
        else:
            all_fields = None
        results = self.analyzer.analyze(
            text=sample.full_text,
            entities=self.entities,
            language="en",
            all_fields=all_fields,
        )
        starts = []
        ends = []
        scores = []
        tags = []
        #
        for res in results:
            #
            if res.score >= self.score_threshold:
                starts.append(res.start)
                ends.append(res.end)
                tags.append(res.entity_type)
                scores.append(res.score)
        #
        response_tags = span_to_tag(
            scheme=self.labeling_scheme,
            text=sample.full_text,
            start=starts,
            end=ends,
            tokens=sample.tokens,
            scores=scores,
            tag=tags,
        )
        return response_tags


if __name__ == "__main__":
    print("Reading dataset")
    input_samples = read_synth_dataset("../data/synth_dataset.txt")

    print("Preparing dataset by aligning entity names to Presidio's entity names")

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

    updated_samples = ModelEvaluator.align_input_samples_to_presidio_analyzer(
        input_samples, entities_mapping
    )

    flatten = lambda l: [item for sublist in l for item in sublist]
    from collections import Counter

    count_per_entity = Counter(
        [
            span.entity_type
            for span in flatten(
                [input_sample.spans for input_sample in updated_samples]
            )
        ]
    )

    print("Evaluating samples")
    analyzer = PresidioAnalyzerEvaluator(entities_to_keep=count_per_entity.keys())
    evaluated_samples = analyzer.evaluate_all(updated_samples)
    #
    print("Estimating metrics")
    score = analyzer.calculate_score(evaluation_results=evaluated_samples, beta=2.5)
    precision = score.pii_precision
    recall = score.pii_recall
    entity_recall = score.entity_recall_dict
    entity_precision = score.entity_precision_dict
    f = score.pii_f
    errors = score.model_errors
    #
    print("precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F 2.5: {}".format(f))
    print("Precision per entity: {}".format(entity_precision))
    print("Recall per entity: {}".format(entity_recall))
    #
    FN_mistakes = [str(mistake) for mistake in errors if mistake.error_type == "FN"]
    FP_mistakes = [str(mistake) for mistake in errors if mistake.error_type == "FP"]
    other_mistakes = [
        str(mistake) for mistake in errors if mistake.error_type not in ["FN", "FP"]
    ]

    fn = open("../data/fn_30000.txt", "w+", encoding="utf-8")
    fn1 = "\n".join(FN_mistakes)
    fn.write(fn1)
    fn.close()

    fp = open("../data/fp_30000.txt", "w+", encoding="utf-8")
    fp1 = "\n".join(FP_mistakes)
    fp.write(fp1)
    fp.close()

    mistakes_file = open("../data/mistakes_30000.txt", "w+", encoding="utf-8")
    mistakes1 = "\n".join(other_mistakes)
    mistakes_file.write(mistakes1)
    mistakes_file.close()

    from pickle import dump

    dump(evaluated_samples, open("../data/evaluated_samples_30000.pickle", "wb"))
