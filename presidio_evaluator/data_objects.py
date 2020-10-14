from typing import List, Counter, Dict

import spacy
import srsly
from spacy.tokens import Token
from tqdm import tqdm

from presidio_evaluator import span_to_tag, tokenize

SPACY_PRESIDIO_ENTITIES = {
    "ORG": "ORGANIZATION",
    "NORP": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "ORGANIZATION": "ORGANIZATION"
}
PRESIDIO_SPACY_ENTITIES = {
    "ORGANIZATION": "ORG",
    "COUNTRY": "GPE",
    "CITY": "GPE",
    "LOCATION": "GPE",
    "PERSON": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "NATION_MAN": "GPE",
    "NATION_WOMAN": "GPE",
    "NATION_PLURAL": "GPE",
    "NATIONALITY": "GPE",
    "GPE": "GPE",
    "ORG": "ORG",
}


class Span:
    """
    Holds information about the start, end, type nad value
    of an entity in a text
    """

    def __init__(self, entity_type, entity_value, start_position, end_position):
        self.entity_type = entity_type
        self.entity_value = entity_value
        self.start_position = start_position
        self.end_position = end_position

    def intersect(self, other, ignore_entity_type: bool):
        """
        Checks if self intersects with a different Span
        :return: If interesecting, returns the number of
        intersecting characters.
        If not, returns 0
        """

        # if they do not overlap the intersection is 0
        if self.end_position < other.start_position or other.end_position < \
                self.start_position:
            return 0

        # if we are accounting for entity type a diff type means intersection 0
        if not ignore_entity_type and (self.entity_type != other.entity_type):
            return 0

        # otherwise the intersection is min(end) - max(start)
        return min(self.end_position, other.end_position) - max(
            self.start_position,
            other.start_position)

    def __repr__(self):
        return "Type: {}, value: {}, start: {}, end: {}".format(
            self.entity_type, self.entity_value, self.start_position,
            self.end_position)

    def __eq__(self, other):
        return self.entity_type == other.entity_type \
               and self.entity_value == other.entity_value \
               and self.start_position == other.start_position \
               and self.end_position == other.end_position

    def __hash__(self):
        return hash(('entity_type', self.entity_type,
                     'entity_value', self.entity_value,
                     'start_position', self.start_position,
                     'end_position', self.end_position))

    @classmethod
    def from_json(cls, data):
        return cls(**data)


class SimpleSpacyExtensions(object):
    def __init__(self, **kwargs):
        """
        Serialization of Spacy Token extensions.
        see https://spacy.io/api/token#set_extension
        :param kwargs: dictionary of spacy extensions and their values
        """
        self.__dict__.update(kwargs)

    def to_dict(self):
        return self.__dict__


class SimpleToken(object):
    """
    A class mimicking the Spacy Token class, for serialization purposes
    """

    def __init__(self, text, idx, tag_=None,
                 pos_=None,
                 dep_=None,
                 lemma_=None,
                 spacy_extensions: SimpleSpacyExtensions = None,
                 **kwargs):
        self.text = text
        self.idx = idx
        self.tag_ = tag_
        self.pos_ = pos_
        self.dep_ = dep_
        self.lemma_ = lemma_

        # serialization for Spacy extensions:
        if spacy_extensions is None:
            self._ = SimpleSpacyExtensions()
        else:
            self._ = spacy_extensions
        self.params = kwargs

    @classmethod
    def from_spacy_token(cls, token):

        if isinstance(token, SimpleToken):
            return token

        elif isinstance(token, Token):

            if token._ and token._._extensions:
                extensions = list(token._.token_extensions.keys())
                extension_values = {}
                for extension in extensions:
                    extension_values[extension] = token._.__getattr__(extension)
                spacy_extensions = SimpleSpacyExtensions(**extension_values)
            else:
                spacy_extensions = None

            return cls(text=token.text,
                       idx=token.idx,
                       tag_=token.tag_,
                       pos_=token.pos_,
                       dep_=token.dep_,
                       lemma_=token.lemma_,
                       spacy_extensions=spacy_extensions)

    def to_dict(self):
        return {
            "text": self.text,
            "idx": self.idx,
            "tag_": self.tag_,
            "pos_": self.pos_,
            "dep_": self.dep_,
            "lemma_": self.lemma_,
            "_": self._.to_dict()
        }

    def __repr__(self):
        return self.text

    @classmethod
    def from_json(cls, data):

        if '_' in data:
            data['spacy_extensions'] = \
                SimpleSpacyExtensions(**data['_'])
        return cls(**data)


class InputSample(object):

    def __init__(self, full_text: str, masked: str, spans: List[Span],
                 tokens=[], tags=[],
                 create_tags_from_span=True, scheme="IO", metadata=None, template_id=None):
        """
        Holds all the information needed for evaluation in the
        presidio-evaluator framework.
        Can generate tags (BIO/BILOU/IO) based on spans

        :param full_text: The raw text of this sample
        :param masked: Masked version of the raw text (desired output)
        :param spans: List of spans for entities
        :param create_tags_from_span: True if tags (tokens+taks) should be added
        :param scheme: IO, BIO/IOB or BILOU. Only applicable if span_to_tag=True
        :param tokens: list of items of type SimpleToken
        :param tags: list of strings representing the label for each token,
        given the scheme
        :param metadata: A dictionary of additional metadata on the sample,
        in the English (or other language) vocabulary
        :param template_id: Original template (utterance) of sample, in case it was generated
        """
        self.full_text = full_text
        self.masked = masked
        self.spans = spans if spans else []
        self.metadata = metadata

        # generated samples have a template from which they were generated
        if not template_id and self.metadata:
            self.template_id = self.metadata.get("Template#")
        else:
            self.template_id = template_id

        if create_tags_from_span:
            tokens, tags = self.get_tags(scheme)
            self.tokens = tokens
            self.tags = tags
        else:
            self.tokens = tokens
            self.tags = tags

    def __repr__(self):
        return "Full text: {}\n" \
               "Spans: {}\n" \
               "Tokens: {}\n" \
               "Tags: {}\n".format(self.full_text, self.spans, self.tokens,
                                   self.tags)

    def to_dict(self):

        return {
            "full_text": self.full_text,
            "masked": self.masked,
            "spans": [span.__dict__ for span in self.spans],
            "tokens": [SimpleToken.from_spacy_token(token).to_dict()
                       for token in self.tokens],
            "tags": self.tags,
            "template_id": self.template_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_json(cls, data):
        if 'spans' in data:
            data['spans'] = [Span.from_json(span) for span in data['spans']]
        if 'tokens' in data:
            data['tokens'] = [SimpleToken.from_json(val) for val in
                              data['tokens']]
        return cls(**data, create_tags_from_span=False)

    def get_tags(self, scheme="IOB"):
        start_indices = [span.start_position for span in self.spans]
        end_indices = [span.end_position for span in self.spans]
        tags = [span.entity_type for span in self.spans]
        tokens = tokenize(self.full_text)

        labels = span_to_tag(scheme=scheme, text=self.full_text, tag=tags,
                             start=start_indices, end=end_indices,
                             tokens=tokens)

        return tokens, labels

    def to_conll(self, translate_tags, scheme="BIO"):

        conll = []
        for i, token in enumerate(self.tokens):
            if translate_tags:
                label = self.translate_tag(self.tags[i], PRESIDIO_SPACY_ENTITIES, ignore_unknown=True)
            else:
                label = self.tags[i]
            conll.append({"text": token.text,
                          "pos": token.pos_,
                          "tag": token.tag_,
                          "Template#": self.metadata['Template#'],
                          "gender": self.metadata['Gender'],
                          "country": self.metadata['Country'],
                          "label": label},
                         )

        return conll

    def get_template_id(self):
        return self.metadata['Template#']

    @staticmethod
    def create_conll_dataset(dataset, translate_tags=True, to_bio=True):
        import pandas as pd

        conlls = []
        i = 0
        for sample in dataset:
            if to_bio:
                sample.bilou_to_bio()
            conll = sample.to_conll(translate_tags=translate_tags)
            for token in conll:
                token['sentence'] = i
                conlls.append(token)
            i += 1

        return pd.DataFrame(conlls)

    def to_spacy(self, entities=None, translate_tags=True):
        entities = [(span.start_position, span.end_position, span.entity_type)
                    for span in self.spans if (entities is None) or (span.entity_type in entities)]
        new_entities = []
        if translate_tags:
            for entity in entities:
                new_tag = self.translate_tag(entity[2], PRESIDIO_SPACY_ENTITIES, ignore_unknown=True)
                new_entities.append((entity[0], entity[1], new_tag))
        else:
            new_entities = entities
        return (self.full_text,
                {"entities": new_entities})

    @classmethod
    def from_spacy(cls, text, annotations, translate_from_spacy=True):
        spans = []
        for annotation in annotations:
            tag = cls.rename_from_spacy_tags([annotation[2]])[0] if translate_from_spacy else annotation[2]
            span = Span(tag, text[annotation[0]: annotation[1]], annotation[0], annotation[1])
            spans.append(span)
        return cls(full_text=text, masked=None, spans=spans)

    @staticmethod
    def create_spacy_dataset(dataset, entities=None, sort_by_template_id=False, translate_tags=True):
        def template_sort(x):
            return x.metadata['Template#']

        if sort_by_template_id:
            dataset.sort(key=template_sort)

        return [sample.to_spacy(entities=entities, translate_tags=translate_tags) for sample in dataset]

    def to_spacy_json(self, entities=None, translate_tags=True):
        token_dicts = []
        for i, token in enumerate(self.tokens):
            if entities:
                tag = self.tags[i] if self.tags[i][2:] in entities else 'O'
            else:
                tag = self.tags[i]

            if translate_tags:
                tag = self.translate_tag(tag, PRESIDIO_SPACY_ENTITIES, ignore_unknown=True)
            token_dicts.append({
                "orth": token.text,
                "tag": token.tag_,
                "ner": tag
            })

        spacy_json_sentence = {
            "raw": self.full_text,
            "sentences": [{
                "tokens": token_dicts
            }
            ]
        }

        return spacy_json_sentence

    def to_spacy_doc(self):
        doc = self.tokens
        spacy_spans = []
        for span in self.spans:
            start_token = [token.i for token in self.tokens if token.idx == span.start_position][0]
            end_token = [token.i for token in self.tokens if token.idx + len(token.text) == span.end_position][0] + 1
            spacy_span = spacy.tokens.span.Span(doc, start=start_token, end=end_token,
                                                label=span.entity_type)
            spacy_spans.append(spacy_span)
        doc.ents = spacy_spans
        return doc

    @staticmethod
    def create_spacy_json(dataset, entities=None, sort_by_template_id=False, translate_tags=True):
        def template_sort(x):
            return x.metadata['Template#']

        if sort_by_template_id:
            dataset.sort(key=template_sort)

        json_str = []
        for i, sample in tqdm(enumerate(dataset)):
            paragraph = sample.to_spacy_json(entities=entities, translate_tags=translate_tags)
            json_str.append({
                "id": i,
                "paragraphs": [paragraph]
            })

        return json_str

    @staticmethod
    def translate_tags(tags, dictionary, ignore_unknown):
        """
        Translates entity types from one set to another
        :param tags: list of entities to translate, e.g. ["LOCATION","O","PERSON"]
        :param dictionary: Dictionary of old tags to new tags
        :param ignore_unknown: Whether to put "O" when word not in dictionary or keep old entity type
        :return: list of translated entities
        """
        new_tags = []
        for tag in tags:
            new_tags.append(InputSample.translate_tag(tag, dictionary, ignore_unknown))

        return new_tags

    @staticmethod
    def translate_tag(tag, dictionary, ignore_unknown):
        has_prefix = len(tag) > 2 and tag[1] == '-'
        no_prefix = tag[2:] if has_prefix else tag
        if no_prefix in dictionary.keys():
            return tag[:2] + dictionary[no_prefix] if has_prefix else dictionary[no_prefix]
        else:
            if ignore_unknown:
                return "O"
            else:
                return tag

    def bilou_to_bio(self):
        new_tags = []
        for tag in self.tags:
            new_tag = tag
            has_prefix = len(tag) > 2 and tag[1] == '-'
            if has_prefix:
                if tag[0] == 'U':
                    new_tag = 'B' + tag[1:]
                elif tag[0] == 'L':
                    new_tag = 'I' + tag[1:]
            new_tags.append(new_tag)

        self.tags = new_tags


    @staticmethod
    def rename_from_spacy_tags(spacy_tags, ignore_unknown=False):
        return InputSample.translate_tags(spacy_tags, SPACY_PRESIDIO_ENTITIES, ignore_unknown=ignore_unknown)

    @staticmethod
    def rename_to_spacy_tags(tags, ignore_unknown=True):
        return InputSample.translate_tags(tags, PRESIDIO_SPACY_ENTITIES, ignore_unknown=ignore_unknown)

    @staticmethod
    def write_spacy_json_from_docs(dataset, filename="spacy_output.json"):
        docs = [sample.to_spacy_doc() for sample in dataset]
        srsly.write_json(filename, [spacy.gold.docs_to_json(docs)])

    def to_flair(self):
        for token, i in enumerate(self.tokens):
            return "{} {} {}".format(token, token.pos_, self.tags[i])

    def translate_input_sample_tags(self, dictionary=PRESIDIO_SPACY_ENTITIES, ignore_unknown=True):
        self.tags = InputSample.translate_tags(self.tags, dictionary, ignore_unknown=ignore_unknown)
        for span in self.spans:
            if span.entity_value in PRESIDIO_SPACY_ENTITIES:
                span.entity_value = PRESIDIO_SPACY_ENTITIES[span.entity_value]
            elif ignore_unknown:
                span.entity_value = 'O'

    @staticmethod
    def create_flair_dataset(dataset):
        flair_samples = []
        for sample in dataset:
            flair_samples.append(sample.to_flair())

        return flair_samples


class ModelError:

    def __init__(self, error_type, annotation, prediction, token, full_text, metadata):
        """
        Holds information about an error a model made for analysis purposes
        :param error_type: str, e.g. FP, FN, Person->Address etc.
        :param annotation: ground truth value
        :param prediction: predicted value
        :param token: token in question
        :param full_text: full input text
        :param metadata: metadata on text from InputSample
        """

        self.error_type = error_type
        self.annotation = annotation
        self.prediction = prediction
        self.token = token
        self.full_text = full_text
        self.metadata = metadata

    def __str__(self):
        return "type: {}, " \
               "Annotation = {}, " \
               "prediction = {}, " \
               "Token = {}, " \
               "Full text = {}, " \
               "Metadata = {}".format(self.error_type,
                                      self.annotation,
                                      self.prediction,
                                      self.token,
                                      self.full_text,
                                      self.metadata)

    def __repr__(self):
        return r"<ModelError {{0}}>".format(self.__str__())


class EvaluationResult(object):
    def __init__(self, results: Counter, model_errors: List[ModelError], text: str = None):
        """
        Holds the output of a comparison between ground truth and predicted
        :param results: List of objects of type Counter
        with structure {(actual, predicted) : count}
        :param model_errors: List of ModelError
        :param text: sample's full text (if used for one sample)
        :type results: Counter
        :type model_errors : List[ModelError]
        :type text: object
        """
        self.results = results
        self.model_errors = model_errors
        self.text = text

        self.pii_recall = None
        self.pii_precision = None
        self.pii_f = None
        self.entity_recall_dict = None
        self.entity_precision_dict = None

    def print(self):
        recall_dict = self.entity_recall_dict
        precision_dict = self.entity_precision_dict

        recall_dict["PII"] = self.pii_recall
        precision_dict["PII"] = self.pii_precision

        entities = recall_dict.keys()
        recall = recall_dict.values()
        precision = precision_dict.values()

        row_format = "{:>30}{:>30.2%}{:>30.2%}"
        header_format = "{:>30}" * 3
        print(header_format.format(*("Entity", "Precision", "Recall")))
        for entity, precision, recall in zip(entities, precision, recall):
            print(row_format.format(entity, precision, recall))

        print("PII F measure: {}".format(self.pii_f))

