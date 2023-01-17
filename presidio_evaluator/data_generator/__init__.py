from pathlib import Path

from . import raw_data

_raw_data_path = raw_data.__path__
if not hasattr(_raw_data_path, '__getitem__'):
    _raw_data_path = _raw_data_path._path
raw_data_dir = Path(_raw_data_path[0])

from .presidio_data_generator import SentenceFaker, PresidioSentenceFaker
from .presidio_pseudonymize import PresidioPseudonymization


def read_synth_dataset():
    raise DeprecationWarning(
        "read_synth_dataset is deprecated. " "Please use InputSample.read_dataset_json"
    )


__all__ = ["SentenceFaker", "PresidioSentenceFaker", "PresidioPseudonymization", "read_synth_dataset",
           "raw_data_dir"]
