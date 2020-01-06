import random
import os
from pathlib import Path


class OrgNameGenerator:
    def __init__(self, company_name_file_path="raw_data/organizations.csv"):
        self.companies = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = Path(dir_path, company_name_file_path)

        with open(str(file_path)) as file:
            self.companies = file.read().splitlines()

    def get_organization(self):
        return random.choice(self.companies)
