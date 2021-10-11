import re
from faker import Faker
from typing import List, Optional
from pathlib import Path
from pprint import pprint

class FakeDataGenerator:
    def __init__(
        self,
        locale: Optional[List[str]] = None
    ):
        self.faker = Faker(locale)

    def parse(self, template: str):
        """
        Currently only parses a template using Faker.
        Could use more ways to parse depending on what
        all we need to parse
        """
        try:
            pattern = self.faker.parse(template)
        except AttributeError:
            pattern = f"TODO: {template}!"
        return pattern

    def insert_fake_pii_into_template(self, template):
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

    def insert_fake_data(self,
                         templates_file):

        templates = self.read_template_file(templates_file)
        if templates:
            self.templates = self._prep_templates(templates)
        else:
            self.templates = None

        examples = []
        for template in self.templates:
            examples.append(self.insert_fake_pii_into_template(template))
        return examples

    @staticmethod
    def _prep_templates(raw_templates):
        print("Preparing sample sentences for ingestion")
        # replacement function to convert uppercase letter to lowercase
        def convert_to_lower(match_obj):
            if match_obj.group() is not None:
                return match_obj.group().lower()

        templates = [
            (
                re.sub(r"\[[^()]*\]", convert_to_lower, template.strip())
                  .replace("[", "{"+"{")
                  .replace("]", "}"+"}")
            )
            for template in raw_templates
        ]
        return templates

if __name__ == "__main__":

    template_file_path = Path(__file__).parent / "raw_data" / "templates.txt"
    generator = FakeDataGenerator()
    fake_patterns = generator.insert_fake_data(template_file_path)
    pprint(fake_patterns)