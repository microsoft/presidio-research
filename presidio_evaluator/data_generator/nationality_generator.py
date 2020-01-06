import random
import os
from pathlib import Path
import pandas as pd
import re


class NationalityGenerator:
    def __init__(self, company_name_file_path="raw_data/nationalities.csv"):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = Path(dir_path, company_name_file_path)
        df = pd.read_csv(str(file_path))

        self.df = df

    def get_country(self):
        ## [COUNTRY]
        return NationalityGenerator.capitalizeWords(random.choice(self.df['country'].values))

    def get_nationality(self):
        ## [NATIONALITY]
        return NationalityGenerator.capitalizeWords(random.choice(self.df['nationality'].values))

    def get_nation_woman(self):
        ## [NATION_WOMAN]
        return NationalityGenerator.capitalizeWords(random.choice(self.df['woman'].values))

    def get_nation_man(self):
        ## [NATION_MAN]
        return NationalityGenerator.capitalizeWords(random.choice(self.df['man'].values))

    def get_nation_plural(self):
        ## [NATION_PLURAL]
        return NationalityGenerator.capitalizeWords(random.choice(self.df['plural'].values))

    @staticmethod
    def capitalizeWords(s):
        return re.sub(r'\w+', lambda m: m.group(0).capitalize(), s)
