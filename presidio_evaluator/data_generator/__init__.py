from .presidio_data_generator import PresidioDataGenerator, PresidioFakeRecordGenerator
from .presidio_pseudonymize import PresidioPseudonymization


def read_synth_dataset():
    raise DeprecationWarning(
        "read_synth_dataset is deprecated. " "Please use InputSample.read_dataset_json"
    )


__all__ = ["PresidioDataGenerator", "PresidioFakeRecordGenerator", "PresidioPseudonymization", "read_synth_dataset"]
