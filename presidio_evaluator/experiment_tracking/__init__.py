import os

from .experiment_tracker import ExperimentTracker

try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None


def get_experiment_tracker():
    framework = os.environ.get("tracking_framework", None)
    if not framework or not Experiment:
        return ExperimentTracker()
    elif framework == "comet":
        return Experiment(
            api_key=os.environ.get("API_KEY"),
            project_name=os.environ.get("PROJECT_NAME"),
            workspace=os.environ.get("WORKSPACE"),
        )


__all__ = ["ExperimentTracker", "get_experiment_tracker"]
