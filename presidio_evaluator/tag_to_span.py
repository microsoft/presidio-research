from typing import List, Optional
from spacy.tokens import Doc

from presidio_evaluator import Span


def _to_io(tag):
    if "-" in tag:
        return tag[2:]
    return tag


def tag_to_span(tokens: Doc, tags: List[str]) -> List[Span]:
    """
    Transforms a list of tags to a list of spans, while ignoring labeling scheme.
    :param tokens: List of spaCy tokens
    :param tags: List of tags (e.g. ["O", "B-PER", "I-PER", "U-LOC"])
    :return: List of spans
    """

    # If IO, translate to BIO for easier manipulation
    not_io = any([tag[1] == "-" for tag in tags if len(tag) > 1])
    if not_io:
        tags = [_to_io(tag) for tag in tags]

    if not len(tags):
        return []

    spans: List[Span] = []
    prev_tag = "O"
    prev_token = None
    current_span: Optional[Span] = None
    for token, tag in zip(tokens, tags):
        if prev_tag != "O" and tag != prev_tag:
            # end current before creating a new one
            spans.append(current_span)
            current_span = None

        if tag != "O":
            if prev_tag == "O" or prev_tag != tag:
                # start new span as this tag is a new entity
                current_span = Span(
                    entity_type=tag,
                    entity_value=str(token),
                    start_position=token.idx,
                    end_position=token.idx + len(token),
                )
            else:
                # continue previous span
                current_span.end_position = token.idx + len(token)
                add_space = prev_token.idx + len(prev_token) < token.idx
                current_span.entity_value += " " if add_space else ""
                current_span.entity_value += str(token)

        prev_tag = tag
        prev_token = token

    # Close span if it's still open
    if prev_tag != "O" and current_span:
        spans.append(current_span)

    return spans
