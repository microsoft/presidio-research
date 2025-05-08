from pathlib import Path
import logging
import dspy
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_evaluator import InputSample, split_dataset, Span
from dotenv import load_dotenv

from pydantic import Field, BaseModel as pydBaseModel
from typing import List, Callable, Optional
from presidio_analyzer import RecognizerResult, RemoteRecognizer, AnalysisExplanation

load_dotenv()  # take environment variables

logger = logging.getLogger("presidio_analyzer")


class EntitySpan(pydBaseModel):
    """Holds the information of a single PII entity span."""
    start: int = Field(ge=0, description="Start position of the entity in the text")
    end: int = Field(ge=0, description="End position of the entity in the text")
    entity_type: str = Field(
        description="Type of the entity (e.g., PERSON, LOCATION)", default=""
    )
    entity_value: str = Field(
        description="Value of the entity identified in the text", default=""
    )


class ListOfSpans(pydBaseModel):
    """Holds a list of EntitySpan objects
    which represent the extracted PII entities in the given text."""
    spans: List[EntitySpan] = Field(description="A list of EntitySpan objects")


class PersonalInformationExtractor(dspy.Signature):
    """
    Extract entity types referring to specific personal information (PII), if any, from an input string.
    Output should a list of objects.
    Each object should contain the PII entity type from the following list:
    ["PERSON", "LOCATION",
    "ORGANIZATION", "NORP",
    "DATE_TIME",
    "GPE", "PHONE_NUMBER",
    "EMAIL_ADDRESS", "CREDIT_CARD",
    "IBAN", "DOMAIN_NAME",
    "CRYPTO", "IP_ADDRESS",
    "BANK_ACCOUNT", "DRIVER_LICENSE",
    "SSN", "EMAIL_ADDRESS",
    "AGE","DATE_TIME", "ID"]
    Where:
    - LOCATION is a physical location, address or area (e.g., Silicon Valley, 13 Broadway st. NY 15523)
    - GPE is a geopolitical entity (e.g. Germany, Buenos Aires)
    - NORP is a nationality, religious or political group (e.g. Icelandic, Zoroastrian, Republican)
    - ORGANIZATION is any place a person could work at (e.g. NSA, IBM, Boston Orchestra)
    - ID is any other PII entity that's not covered by the other types but could directly identify an individual

    Objects should contain the corresponding PII text extracted from the input string
    and the start and end indices of the PII in the input string.
    If no PII is found, the output should be an empty list.
    """

    text: str = dspy.InputField(desc="input text")
    extracted_pii: ListOfSpans = dspy.OutputField(
        desc="a list of extracted PII types, values and locations"
    )


def get_optimizer(
    dspy_program: dspy.Module,
    train: List[dspy.Example],
    validation: Optional[List[dspy.Example]] = None,
    metric: Optional[Callable] = None,
) -> dspy.Module:
    available_optimizers = [
        "AvatarOptimizer",
        "BetterTogether",
        "BootstrapFewShot",
        "BootstrapFinetune",
        "COPRO",
        "Ensemble",
        "KNNFewShot",
        "MIPROv2",
        "BootstrapFewShotWithRandomSearch",
        "BootstrapFewShotWithOptuna",
        "LabeledFewShot",
    ]

    #optimizer = dspy.BootstrapFewShot(metric=metric)  # , auto="light", num_threads=1)

    optimizer = dspy.teleprompt.MIPROv2(
        metric=metric,
        max_labeled_demos=4,
        max_bootstrapped_demos=3,
    )

    compiled = optimizer.compile(
        dspy_program.deepcopy(),
        trainset=train,
        valset=validation,
        requires_permission_to_run=False,
    )
    compiled.save(Path(f"../../models/{optimizer.__class__.__name__}_optimized.json"))
    return compiled


def extraction_correctness_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> bool:
    """Check if the predicted spans are correct by comparing them to the example spans.

    :param example: An Example object containing the actual pii spans and the input text
    :param prediction: A Prediction object containing the extracted spans
    """
    predicted_recognizer_result = llm_spans_to_recognizer_results(
        example.text, prediction.extracted_pii.spans
    )
    example_recognizer_result = llm_spans_to_recognizer_results(
        example.text, example.extracted_pii.spans
    )

    predicted_recognizer_result = sorted(
        predicted_recognizer_result, key=lambda x: x.start
    )
    example_recognizer_result = sorted(example_recognizer_result, key=lambda x: x.start)

    if len(predicted_recognizer_result) != len(example_recognizer_result):
        return False

    for pred, actual in zip(predicted_recognizer_result, example_recognizer_result):
        if (
            example.text[pred.start : pred.end]
            != example.text[actual.start : actual.end]
        ):
            return False

    return True


def main():
    lm = dspy.LM(model="openai/gpt-4o-mini")
    # lm = dspy.LM('ollama_chat/phi3')
    dspy.settings.configure(lm=lm)

    dataset_name = "synth_dataset_v2.json"
    dataset = InputSample.read_dataset_json(
        Path(Path.cwd().parent.parent, "data", dataset_name)
    )

    train_set, test_set = split_dataset(dataset, [0.5, 0.5])

    train = prepare_dataset(train_set)[0:200]
    val = prepare_dataset(train_set)[100:200]
    test = prepare_dataset(test_set)[0:200]

    evaluate_correctness = dspy.Evaluate(
        metric=extraction_correctness_metric,
        devset=test[:100],
        num_threads=4,
        display_progress=True,
        display_table=True,
        provide_traceback=True,
    )

    pii_extractor = dspy.ChainOfThought(PersonalInformationExtractor)
    optimized = get_optimizer(
        pii_extractor, train=train, validation=val, metric=extraction_correctness_metric
    )
    evaluate_correctness(optimized, metric=extraction_correctness_metric)


def _list_of_spans_to_spans(entity_spans: ListOfSpans) -> List[Span]:
    spans = []
    for span in entity_spans.spans:
        spans.append(
            Span(
                start_position=span.start,
                end_position=span.end,
                entity_type=span.entity_type,
                entity_value=span.entity_value,
            )
        )
    return spans


def _spans_to_list_of_spans(spans: List[Span]) -> ListOfSpans:
    entity_spans = []
    for span in spans:
        entity_spans.append(
            EntitySpan(
                start=span.start_position,
                end=span.end_position,
                entity_type=span.entity_type,
                entity_value=span.entity_value,
            )
        )
    return ListOfSpans(spans=entity_spans)


def prepare_dataset(dataset: List[InputSample]):
    examples = []
    for sample in dataset:
        list_of_spans = _spans_to_list_of_spans(sample.spans)
        example = dspy.Example(
            text=sample.full_text,
            extracted_pii=list_of_spans,
        ).with_inputs("text")
        examples.append(example)

    return examples


def llm_spans_to_recognizer_results(
    text: str, spans: List[EntitySpan], tolerance: int = 10
) -> List[RecognizerResult]:
    """Translate the predicted spans to a list of RecognizerResult while ensuring the boundaries are correct.

    LLM models (regardless of how well they perform) are struggling to identify the span (start and end index)
    of each entity, while excelling at extracting the text. To ensure that the span is correct,
    this method looks for the entity value in the proposed span with tolerance. If the text is found,
    the span start and end are updated. If not, the entity value is searched for in the full text.

    :param text: The input text
    :param spans: A list of EntitySpan objects
    :param tolerance: Padding on each side when searching for the entity value in the text

    """
    recognizer_results = []
    for span in spans:
        if text[span.start : span.end] == span.entity_value:
            # Original span is correct
            recognizer_results.append(
                RecognizerResult(
                    start=span.start,
                    end=span.end,
                    entity_type=span.entity_type,
                    score=0.9,
                )
            )
            continue

        # Define the broader search range
        search_start = max(0, span.start - tolerance)
        search_end = min(len(text), span.end + tolerance)
        search_text = text[search_start:search_end]

        # Find the true location of the entity value within the broader range
        true_start = search_text.find(span.entity_value)

        if true_start != -1:
            # Adjust the true start and end positions relative to the full text
            adjusted_start = search_start + true_start
            adjusted_end = adjusted_start + len(span.entity_value)
        else:
            # If not found, search in the entire text
            true_start = text.find(span.entity_value)
            if true_start != -1:
                adjusted_start = true_start
                adjusted_end = adjusted_start + len(span.entity_value)
            else:
                # If not found anywhere, retain the original span
                recognizer_results.append(
                    RecognizerResult(
                        start=span.start,
                        end=span.end,
                        entity_type=span.entity_type,
                        score=0.9,
                    )
                )
                continue

        # Create the updated span
        updated_span = RecognizerResult(
            start=adjusted_start,
            end=adjusted_end,
            entity_type=span.entity_type,
            score=0.9,
        )
        recognizer_results.append(updated_span)

    return recognizer_results


class DspyRecognizer(RemoteRecognizer):
    def __init__(self, supported_entities: Optional[List[str]] = None):
        if not supported_entities:
            logger.info("No supported entities provided. Using default entities.")
            supported_entities = [
                "PERSON",
                "LOCATION",
                "ADDRESS",
                "ORGANIZATION",
                "NORP",
                "DATE_TIME",
                "GPE",
                "PHONE_NUMBER",
                "EMAIL_ADDRESS",
                "CREDIT_CARD",
                "IBAN",
                "DOMAIN_NAME",
                "CRYPTO",
                "IP_ADDRESS",
                "BANK_ACCOUNT",
                "DRIVER_LICENSE",
                "SSN",
                "EMAIL_ADDRESS",
                "AGE",
                "DATE_TIME",
                "ID",
            ]

        super().__init__(
            supported_entities=supported_entities,
            supported_language="en",
            name="DspyRecognizer",
            version="0.1",
        )
        lm = dspy.LM(model="openai/gpt-4o-mini")
        dspy.settings.configure(lm=lm)

        self.pii_extractor = dspy.ChainOfThought(PersonalInformationExtractor)
        self.pii_extractor.load("../models/MipROv2_optimized.json")

    def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts):
        prediction = self.pii_extractor(text=text)
        extracted_pii = prediction.extracted_pii

        results = llm_spans_to_recognizer_results(text, extracted_pii.spans)
        updated_results = []
        for result in results:
            if entities and result.entity_type not in entities:
                continue
            result.analysis_explanation = AnalysisExplanation(
                recognizer=self.name,
                original_score=result.score,
                textual_explanation=f"Identified as {result.entity_type} "
                                    f"by openai/gpt-4o-mini using dspy/ChainOfThought"
            )
            updated_results.append(result)
        return results

    def get_supported_entities(self) -> List[str]:
        return self.supported_entities


if __name__ == "__main__":
    main()
