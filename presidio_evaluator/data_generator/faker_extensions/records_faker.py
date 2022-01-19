from typing import Union, Dict, List

from faker import Faker
import pandas as pd

from presidio_evaluator.data_generator.faker_extensions import RecordGenerator


class RecordsFaker(Faker):
    def __init__(self, records: Union[pd.DataFrame, List[Dict]], **kwargs):
        if isinstance(records, pd.DataFrame):
            records = records.to_dict(orient="records")

        record_generator = RecordGenerator(records=records)
        super().__init__(generator=record_generator, **kwargs)
