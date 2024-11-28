from dataclasses import dataclass
import dataclasses
import json
from pathlib import Path
from typing import Optional, List, Union
from collections import Counter
from typing import Dict


@dataclass(eq=True)
class FakerSpan:
    """FakerSpan holds the start, end, value and type of every element replaced."""

    value: str
    start: int
    end: int
    type: str

    def __repr__(self):
        return json.dumps(dataclasses.asdict(self))


@dataclass()
class FakerSpansResult:
    """FakerSpansResult holds the full fake sentence, the original template
    and a list of spans for each element replaced."""

    fake: str
    spans: List[FakerSpan]
    template: Optional[str] = None
    template_id: Optional[int] = None
    sample_id: Optional[int] = None

    def __str__(self):
        return self.fake

    def __repr__(self):
        return json.dumps(dataclasses.asdict(self))

    def toJSON(self):
        spans_dict = json.dumps([dataclasses.asdict(span) for span in self.spans])
        return json.dumps(
            {
                "fake": self.fake,
                "spans": spans_dict,
                "template": self.template,
                "template_id": self.template_id,
                "sample_id": self.sample_id,
            }
        )

    @classmethod
    def fromJSON(cls, json_string):
        """Load a single FakerSpansResult from a JSON string."""
        json_dict = json.loads(json_string)
        converted_spans = []
        for span_dict in json.loads(json_dict["spans"]):
            converted_spans.append(FakerSpan(**span_dict))
        json_dict["spans"] = converted_spans
        return cls(**json_dict)

    @classmethod
    def count_entities(cls, fake_records: List["FakerSpansResult"]) -> Counter:
        """Count frequency of entity types in a list of FakerSpansResult."""
        count_per_entity_new = Counter()
        for record in fake_records:
            for span in record.spans:
                count_per_entity_new[span.type] += 1
        return count_per_entity_new.most_common()

    @classmethod
    def load_dataset_from_file(
        cls, filename: Union[Path, str]
    ) -> List["FakerSpansResult"]:
        """Load a dataset of FakerSpansResult from a JSON file."""
        with open(filename, "r", encoding="utf-8") as f:
            return [cls.fromJSON(line) for line in f.readlines()]

    @classmethod
    def update_entity_types(
        cls, dataset: List["FakerSpansResult"], entity_mapping: Dict[str, str]
    ):
        """Replace entity types using a translator dictionary."""
        for sample in dataset:
            # update entity types on spans
            for span in sample.spans:
                span.type = entity_mapping[span.type]
            # update entity types on the template string
            for key, value in entity_mapping.items():
                sample.template = sample.template.replace(
                    "{{" + key + "}}", "{{" + value + "}}"
                )
