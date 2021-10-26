from typing import Dict, Optional, List, Any

from faker.providers import DynamicProvider
from faker.generator import _re_token
from presidio_evaluator.data_generator.faker_extensions import Span, SpanGenerator


class RecordGenerator(SpanGenerator):
    """Prioritizes the sampling of values found in input records,
    in order to allow semantically similar elements to be drawn together.

    For example, for the input template
    "My first name is {{name}} and my email is {{email}}",
    assuming the user added a list of records containing name and email
    (e.g. `[{"name": "Stephane Smith", "email": "stephane.smith@gmail.com"}]`),
    this generator will sample the name and email from the same record
    and not independently.
    In case the template contains entities not found in the record,
    the generator will use regular Faker providers.
    In case the template contains the same type multiple times,
    the first would be taken from the record and the next ones
    from regular Faker providers.

    :example:
    >>>from faker import Faker
    >>>from presidio_evaluator.data_generator.faker_extensions import RecordGenerator

    >>>records = [
    >>>     {"name": "Alan", "email": "alan@a.com"},
    >>>     {"name": "Barry", "email": "barry@b.com"},
    >>>     {"name": "Cynthia", "email": "cynthia@c.com"},
    >>>     ]

    >>>generator = RecordGenerator(records=records)
    >>>faker = Faker(generator=generator)

    >>># Sample fake values from the same record
    >>>faker.parse("I'm {{name}} and my email is {{email}}.")

    I'm Alan and my email is alan@a.com.

    >>># Using more than one type will use regular Faker providers:
    >>>faker.parse("{{name}}, {{name}} and {{name}} will email {{email}}.")

    Cynthia, Manuel Gonzales and Jillian Riley will email cynthia@c.com.

    >>># Sample from a Pandas DataFrame
    >>>import pandas as pd
    >>>df = pd.DataFrame({"name":["a","b","c"],"email":["a@a","b@b","c@c"]}) # or read from file
    >>>records = df.to_dict(orient="records")
    >>>generator = RecordGenerator(records=records)
    >>>faker = Faker(generator=generator)
    >>>faker.parse("I'm {{name}} and my email is {{email}}")

    I'm a and my email is a@a
    """

    def __init__(self, records: Optional[List[Dict]] = None):
        super().__init__()
        self.records = records

        if self.records:
            for record in self.records:
                if not isinstance(record, Dict):
                    raise TypeError("Each element should be of type Dict")

        # Use an internal provider to sample from the input elements
        self.dynamic_record_provider = DynamicProvider(
            provider_name="", elements=records, generator=self
        )

    def _get_random_record(self):
        return self.dynamic_record_provider.get_random_value().copy()

    def _match_to_span(self, text: str, **kwargs) -> List[Span]:
        """Adds logic for sampling from input records if possible"""
        matches = _re_token.finditer(text)
        record = self._get_random_record()

        results: List[Span] = []
        for match in matches:
            formatter = match.group()[2:-2]
            stripped = formatter.strip()

            value = self.format(formatter=stripped, record=record)
            if stripped in record:
                del record[stripped]  # Remove in order not to sample twice

            results.append(
                Span(
                    type=formatter,
                    start=match.start(),
                    end=match.end(),
                    value=value,
                )
            )

        return results

    def format(self, formatter: str, *args: Any, **kwargs: Any) -> str:
        record = kwargs.get("record")
        if not record or (formatter not in record):
            return super().format(formatter)

        return record[formatter]