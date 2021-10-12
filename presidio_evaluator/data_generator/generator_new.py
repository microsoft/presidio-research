import re
from faker import Faker
from typing import List, Optional
from pathlib import Path
from pprint import pprint

class FakeDataGenerator:
    def __init__(
        self,
        custom_faker: Faker = None,
        locale: Optional[List[str]] = None
    ):
        """
        Fake data generator.
        Leverages Faker to create fake PII entities into predefined templates of structure: a b c {{PII}} d e f,
        e.g. "My name is {{first_name}}."
        :param custom_faker: A Faker object provided by the user
        :param locale: A locale object to create our own Faker instance if a custom one was not provided.
        """
        if custom_faker:
            self.faker = custom_faker
        else:
            self.faker = Faker(locale)

    def parse(self, template: str):
        """
        Currently only parses a template using Faker.
        Could use more ways to parse depending on what
        all we need to parse
        Args:
            template: str with token(s) that needs to be replaced by fake PII
            Examples:
            1. "My name is {{first_name_female}} {{last_name}}".
            2. "I want to increase limit on my card # {{credit_card_number}}
                for certain duration of time. is it possible?"
        Returns:
            A sentence with fake PII in it [or] an Exception.
        """
        try:
            pattern = self.faker.parse(template)
            return pattern
        except Exception as err:
            raise AttributeError(f"{err}! You could create a new provider!")

    def generate_fake_pii_for_template(self, template):
        """
        This function replaces known PII {{tokens}} in a template sentence
        with a fake value for each token and returns a sentence with fake PII.

        Args:
            template: str with token(s) that needs to be replaced by fake PII
            Examples:
            1. "My name is {{first_name_female}} {{last_name}}".
            2. "I want to increase limit on my card # {{credit_card_number}}
                for certain duration of time. is it possible?"

        Returns:
            Fake sentence.

        """
        pattern = self.parse(template)
        return pattern

    @staticmethod
    def read_template_file(templates_file):
        with open(templates_file) as f:
            return f.readlines()

    @staticmethod
    def _prep_templates(raw_templates):
        print("Preparing sample sentences for ingestion")
        def make_lower_case(match_obj):
            if match_obj.group() is not None:
                return match_obj.group().lower()

        templates = [(
            re.sub(r'\[.*?\]', make_lower_case, template.strip())
              .replace("[", "{"+"{")
              .replace("]", "}"+"}")
        )
        for template in raw_templates
        ]

        return templates

    def generate_fake_data(self,
                           templates_file):
        """
        Generates fake PII data whenever it encounters known faker entities in a template.
        Args:
            templates_file: A path to a Faker-style template file
        Returns:
            List: Example Sentences with fake values for PII entities in templates
        """

        templates = self.read_template_file(templates_file)

        if templates:
            self.templates = self._prep_templates(templates)
        else:
            self.templates = None

        examples = []
        for template in self.templates:
            examples.append(self.generate_fake_pii_for_template(template))
        return examples

if __name__ == "__main__":

    template_file_path = Path(__file__).parent / "raw_data" / "faker_templates.txt"
    custom_faker = Faker('fa_IR')
    generator = FakeDataGenerator(custom_faker=None, locale='jp_JP')
    fake_patterns = generator.generate_fake_data(template_file_path)
    pprint(fake_patterns)
