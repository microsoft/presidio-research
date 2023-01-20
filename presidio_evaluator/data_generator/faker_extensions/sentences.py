import random
from typing import Optional, List, Union, Dict, Any

import faker
import numpy as np
import pandas as pd
from faker import Faker
from faker.generator import _re_token
from faker.providers import BaseProvider, DynamicProvider

from presidio_evaluator.data_generator.faker_extensions import SpanGenerator, FakerSpansResult as FakeSentenceResult, \
    FakerSpan


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

    >>># Return spans of fake values
    >>>res = faker.parse("I'm {{name}} and my email is {{email}}",add_spans=True)

    {"fake": "I'm c and my email is c@c",
     "spans": "[{\"value\": \"c@c\", \"start\": 22, \"end\": 25, \"type\": \"email\"},
     {\"value\": \"c\", \"start\": 4, \"end\": 5, \"type\": \"name\"}]"
     }

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

    def _match_to_span(self, text: str, **kwargs) -> List[FakerSpan]:
        """Adds logic for sampling from input records if possible."""
        matches = _re_token.finditer(text)
        # Sample one record (Dict containing fake values)
        record = self._get_random_record()

        results: List[FakerSpan] = []
        for match in matches:
            formatter = match.group()[2:-2]
            stripped = formatter.strip()

            value = str(self.format(formatter=stripped, record=record))
            if stripped in record:
                del record[stripped]  # Remove in order not to sample twice

            results.append(
                FakerSpan(
                    type=formatter,
                    start=match.start(),
                    end=match.end(),
                    value=value,
                )
            )

        return results

    def format(self, formatter: str, *args: Any, **kwargs: Any) -> str:
        """Fill in fake data. If the input record has the requested entity, return its value."""
        record = kwargs.get("record")
        if not record or not record.get(
                formatter
        ):  # type not in record, go to default faker
            return super().format(formatter)

        return record[formatter]


class RecordsFaker(Faker):
    def __init__(self, records: Union[pd.DataFrame, List[Dict]], **kwargs):
        if isinstance(records, pd.DataFrame):
            records = records.to_dict(orient="records")

        record_generator = RecordGenerator(records=records)
        super().__init__(generator=record_generator, **kwargs)


class SentenceFaker:
    def __init__(
            self,
            custom_faker: Optional[Faker] = None,
            locale: Optional[List[str]] = None,
            lower_case_ratio: float = 0.05,
    ):
        """
        Leverages Faker to create fake PII entities into predefined templates of structure: a b c {{PII}} d e f,
        e.g. "My name is {{first_name}}."
        :param custom_faker: A Faker object provided by the user
        :param locale: A locale object to create our own Faker instance if a custom one was not provided.
        :param lower_case_ratio: Percentage of names that should start with lower case

        :example:

        >>>from presidio_evaluator.data_generator.faker_extensions.sentences import SentenceFaker

        >>>template = "I just moved to {{city}} from {{country}}"
        >>>fake_sentence_result = SentenceFaker().parse(template)
        >>>print(fake_sentence_result.fake)
        I just moved to North Kim from Ukraine
        >>>print(fake_sentence_result.spans)
        [{"value": "Ukraine", "start": 31, "end": 38, "type": "country"}, {"value": "North Kim", "start": 16, "end": 25, "type": "city"}]
        """
        if custom_faker and locale:
            raise ValueError("If a custom faker is passed, it's expected to have its locales loaded")

        if custom_faker:
            self.faker = custom_faker
        else:
            generator = (
                SpanGenerator()
            )  # To allow PresidioDataGenerator to return spans and not just strings
            self.faker = Faker(local=locale, generator=generator)
        self.lower_case_ratio = lower_case_ratio

    def parse(
            self, template: str, template_id: Optional[int] = None, add_spans: Optional[bool] = True
    ) -> Union[FakeSentenceResult, str]:
        """
        This function replaces known PII {{tokens}} in a template sentence
        with a fake value for each token and returns a sentence with fake PII.

        Examples:
            1. "My name is {{first_name_female}} {{last_name}}".
            2. "I want to increase limit on my card # {{credit_card_number}}
                for certain duration of time. is it possible?"


        :param template: str with token(s) that needs to be replaced by fake PII
        :param template_id: The identifier of the specific template
        :param add_spans: Whether to return the spans or just the fake text

        :returns: Fake sentence.

        """
        try:
            if isinstance(self.faker.factories[0], SpanGenerator):
                fake_pattern = self.faker.parse(
                    template, add_spans=add_spans, template_id=template_id
                )
            else:
                fake_pattern = self.faker.parse(template)
            if random.random() < self.lower_case_ratio:
                fake_pattern = self._lower_pattern(fake_pattern)
            return fake_pattern
        except Exception as err:
            raise AttributeError(
                f'Failed to generate fake data based on template "{template}".'
                f"You might need to add a new Faker provider! "
                f"{err}"
            )

    @staticmethod
    def _lower_pattern(pattern: Union[str, FakeSentenceResult]):
        if isinstance(pattern, str):
            return pattern.lower()
        elif isinstance(pattern, FakeSentenceResult):
            pattern.fake = pattern.fake.lower()
            for span in pattern.spans:
                span.value = str(span.value).lower()
            return pattern

    def seed(self, seed_value=42):
        Faker.seed(seed_value)
        self.faker.seed_instance(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)

    def add_provider_alias(self, provider_name: str, new_name: str) -> None:
        """
        Adds a copy of a provider, with a different name
        :param provider_name: Name of original provider
        :param new_name: New name
        :example:
        >>>add_provider_alias(provider_name="name", new_name="person")
        >>>self.faker.person()
        """
        original = getattr(self.faker, provider_name)

        new_provider = BaseProvider(self.faker)
        setattr(new_provider, new_name, original)
        self.faker.add_provider(new_provider)
