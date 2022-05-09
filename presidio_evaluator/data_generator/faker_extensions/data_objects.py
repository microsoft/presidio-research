from dataclasses import dataclass
import dataclasses
import json
from typing import Optional, List


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
            }
        )
