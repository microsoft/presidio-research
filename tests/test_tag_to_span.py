import pytest
import spacy

from presidio_evaluator import tag_to_span


@pytest.fixture()
def spacy_en_core_web_sm():
    return spacy.load("en_core_web_sm")


@pytest.mark.parametrize(
    "text, tags, ents",
    [
        ("Dan  is my name. ", ["B-name", "O", "O", "O", "O"], [[0, 1, "name"]]),
        ("Dan Brown lives here", ["B-name", "I-name", "O", "O", "O"], [[0, 2, "name"]]),
        ("Dan Dan Dan", ["B-name", "B-name", "B-name"], [[0, 3, "name"]]),
        ("Nothing to look for", ["O", "O", "O", "O"], []),
        ("It's Dan", ["O", "O", "B-name"], [[2, 3, "name"]]),  # BIO tagging scheme
        ("It's Dan", ["O", "O", "name"], [[2, 3, "name"]]),  # IO tagging scheme
        ("It's Dan", ["O", "O", "B-name"], [[2, 3, "name"]]),  # wrong tagging scheme
        (
            "It's Dan Washington",
            ["O", "O", "U-name", "U-loc"],
            [[2, 3, "name"], [3, 4, "loc"]],
        ),
        (
            "Dan is Washington",
            ["U-name", "O", "U-loc"],
            [[0, 1, "name"], [2, 3, "loc"]],
        ),
    ],
)
def test_tag_to_span(spacy_en_core_web_sm, text, tags, ents):
    doc = spacy_en_core_web_sm(text)
    spacy_ents = [spacy.tokens.Span(doc, ent[0], ent[1], label=ent[2]) for ent in ents]
    doc.ents = spacy_ents

    spans = tag_to_span(doc, tags)

    assert len(spans) == len(ents)
    for span, spacy_span in zip(spans, spacy_ents):
        assert span.start_position == spacy_span.start_char
        assert span.end_position == spacy_span.end_char
        assert span.entity_type == spacy_span.label_
        assert span.entity_value == str(spacy_span)
