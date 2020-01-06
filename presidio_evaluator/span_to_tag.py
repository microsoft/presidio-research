from collections import namedtuple
from typing import List

import spacy

loaded_spacy = {}


def get_spacy(loaded_spacy=loaded_spacy, model_version="en_core_web_lg"):
    if model_version not in loaded_spacy:
        disable = ['vectors', 'textcat', 'ner']
        print("loading model {}".format(model_version))
        loaded_spacy[model_version] = spacy.load(model_version, disable=disable)
    return loaded_spacy[model_version]


def tokenize(text, model_version="en_core_web_lg"):
    return get_spacy(model_version=model_version)(text)


def _get_detailed_tags(scheme, cur_tags):
    """
    Replaces IO tags (e.g. PERSON PERSON) with IOB/BIO/BILOU tags
    :param cur_tags:
    :param scheme:
    :return:
    """

    if all([tag == 'O' for tag in cur_tags]):
        return cur_tags

    return_tags = []
    if len(cur_tags) == 1:
        if scheme == "BILOU":
            return_tags.append("U-{}".format(cur_tags[0]))
        else:
            return_tags.append("I-{}".format(cur_tags[0]))
    elif len(cur_tags) > 0:
        tg = cur_tags[0]
        for j in range(0, len(cur_tags)):
            if j == 0:
                return_tags.append("B-{}".format(tg))
            elif j == len(cur_tags) - 1:
                if scheme == "BILOU":
                    return_tags.append("L-{}".format(tg))
                else:
                    return_tags.append("I-{}".format(tg))
            else:
                return_tags.append("I-{}".format(tg))
    return return_tags


def _sort_spans(start, end, tag, score):
    if len(start) > 0:
        tpl = [(a, b, c, d) for a, b, c, d in sorted(zip(start, end, tag, score), key=lambda pair: pair[0])]
        start, end, tag, score = [[x[i] for x in tpl] for i in range(len(tpl[0]))]
    return start, end, tag, score


def _handle_overlaps(start, end, tag, score):
    start, end, tag, score = _sort_spans(start, end, tag, score)
    if len(start) == 0:
        return start, end, tag, score
    max_end = max(end)
    index = min(start)
    number_of_spans = len(start)
    i = 0
    while i < number_of_spans-1:
        for j in range(i+1,number_of_spans):
            # Span j intersects with span i
            if start[i] <= start[j] <= end[i]:
                # i's score is higher, remove intersecting part
                if score[i] > score[j]:
                    # j is contained within i but has lower score, remove
                    if start[i] >= end[j] >= end[i]:
                        score[j] = 0
                    # else, j continues after i ended:
                    else:
                        start[j] = end[i] + 1
                # j's score is higher, break i
                else:
                    # If i finishes after j ended, split i
                    if end[j] < end[i]:
                        # create new span at the end
                        start.append(end[j] + 1)
                        end.append(end[i])
                        score.append(score[i])
                        tag.append(tag[i])
                        number_of_spans += 1
                        # truncate the current i to end at start(j)
                        end[i] = start[j] - 1
                    # else, i finishes before j ended. truncate i
                    else:
                        end[i] = start[j] - 1

        i += 1
    start, end, tag, score = _sort_spans(start, end, tag, score)
    return start, end, tag, score


def span_to_tag(scheme: str,
                text: str,
                start: List[int],
                end: List[int],
                tag: List[str],
                scores: List[float] = None,
                tokens: List[spacy.tokens.Token] = None,
                io_tags_only=False) -> List[str]:
    """
    Turns a list of start and end values with corresponding labels, into a NER
    tagging (BILOU,BIO/IOB)
    :param scheme: labeling scheme, either BILOU, BIO/IOB or IO
    :param text: input text
    :param tokens: text tokenized to tokens
    :param start: list of indices where entities in the text start
    :param end: list of indices where entities in the text end
    :param tag: list of entity names
    :param scores: score of tag (confidence)
    :param io_tags_only: Whether to return only I and O tags
    :return: list of strings, representing either BILOU or BIO for the input
    """

    if not scores:
        # assume all scores are of equal weight
        scores = [0.5 for start in start]

    start, end, tag, scores = _handle_overlaps(start, end, tag, scores)

    if not tokens:
        tokens = tokenize(text)

    io_tags = []
    for token in tokens:
        found = False
        for span_index in range(0, len(start)):
            if start[span_index] <= token.idx < end[span_index]:
                io_tags.append(tag[span_index])
                found = True
                break

        if not found:
            io_tags.append("O")

    if io_tags_only or scheme == "IO":
        return io_tags

    # Set tagging based on scheme (BIO/IOB or BILOU)
    current_tag = ""
    span_index = 0
    changes = []
    for io_tag in io_tags:
        if io_tag != current_tag:
            changes.append(span_index)
        span_index += 1
        current_tag = io_tag
    changes.append(len(io_tags))

    new_return_tags = []
    for i in range(len(changes) - 1):
        new_return_tags.extend(
            _get_detailed_tags(scheme=scheme,
                               cur_tags=io_tags[changes[i]:changes[i + 1]]))

    return new_return_tags
