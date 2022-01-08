from typing import List

import spacy

from presidio_evaluator import InputSample, Span, span_to_tag, tokenize

try:
    import spacy_stanza
    import stanza
except ImportError:
    print("stanza and spacy_stanza are not installed")
from presidio_evaluator.models import SpacyModel


class StanzaModel(SpacyModel):
    def __init__(
        self,
        model: spacy.language.Language = None,
        model_name: str = "en",
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        labeling_scheme: str = "BIO",
        translate_to_spacy_entities=True,
    ):
        if not model and not model_name:
            raise ValueError("Either model_name or model object must be supplied")
        if not model:
            model = spacy_stanza.load_pipeline(
                model_name,
                processors="tokenize,pos,lemma,ner",
            )

        super().__init__(
            model=model,
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
            translate_to_spacy_entities=translate_to_spacy_entities,
        )

    def predict(self, sample: InputSample) -> List[str]:
        if self.translate_to_spacy_entities:
            sample.translate_input_sample_tags()

        doc = self.model(sample.full_text)
        if doc.ents:
            tags, texts, start, end = zip(
                *[(s.label_, s.text, s.start_char, s.end_char) for s in doc.ents]
            )

            # Stanza tokens might not be consistent with spaCy's tokens.
            # Use spacy tokenization and not stanza to maintain consistency with other models:
            if not sample.tokens:
                sample.tokens = tokenize(sample.full_text)

            # Create tags (label per token) based on stanza spans and spacy tokens
            tags = span_to_tag(
                scheme=self.labeling_scheme,
                text=sample.full_text,
                start=start,
                end=end,
                tag=tags,
                tokens=sample.tokens
            )
        else:
            tags = ["O" for _ in range(len(sample.tokens))]

        if len(tags) != len(sample.tokens):
            print("mismatch between input tokens and new tokens")

        return tags
