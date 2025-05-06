import pytest
from unittest.mock import MagicMock, patch
from presidio_evaluator.evaluation.plotter import Plotter
from presidio_evaluator.evaluation import EvaluationResult
from pathlib import Path
import pandas as pd

@pytest.fixture
def mock_evaluation_result():
    return EvaluationResult(
        entity_recall_dict={"PERSON": 0.8, "LOCATION": 0.7},
        entity_precision_dict={"PERSON": 0.9, "LOCATION": 0.6},
        n_dict={"PERSON": 50, "LOCATION": 30},
        pii_recall=0.75,
        pii_precision=0.85,
        pii_f=0.8,
        n=80,
        model_errors=[]
    )

@pytest.mark.parametrize("output_path", [None, Path("/tmp/plots")])
def test_plot_scores(mock_evaluation_result, output_path):
    plotter = Plotter(results=mock_evaluation_result, model_name="TestModel")

    with patch("pathlib.Path.mkdir") as mock_mkdir, patch("plotly.graph_objs.Figure.write_image") as mock_write_image, patch("plotly.graph_objs.Figure.show") as mock_show:
        plotter.plot_scores(output_path=output_path)

        if output_path:
            # If output_path is provided, ensure plots are saved to the correct paths
            mock_mkdir.assert_called_once()
            expected_recall_path = output_path / "TestModel-recall.png"
            expected_precision_path = output_path / "TestModel-precision.png"
            expected_fbeta_path = output_path / "TestModel-fbeta.png"

            mock_write_image.assert_any_call(expected_recall_path)
            mock_write_image.assert_any_call(expected_precision_path)
            mock_write_image.assert_any_call(expected_fbeta_path)
            assert mock_write_image.call_count == 3  # Three plots: recall, precision, F-beta
        else:
            # If output_path is not provided, ensure plots are shown instead of saved
            mock_show.assert_called()
            assert mock_show.call_count == 3  # Three plots: recall, precision, F-beta


@pytest.mark.parametrize("output_path", [None, Path("/tmp/plots")])
def test_plot_confusion_matrix(mock_evaluation_result, output_path):
    plotter = Plotter(results=mock_evaluation_result, model_name="TestModel")
    entities = ["PERSON", "LOCATION"]
    confmatrix = [[40, 10], [5, 25]]  # Example confusion matrix

    with patch("pathlib.Path.mkdir") as mock_mkdir, patch("plotly.graph_objs.Figure.write_image") as mock_write_image, patch("plotly.graph_objs.Figure.show") as mock_show:
        plotter.plot_confusion_matrix(entities=entities, confmatrix=confmatrix, output_folder=output_path)

        if output_path:
            # If output_path is provided, ensure the confusion matrix is saved
            mock_mkdir.assert_called_once()
            expected_path = output_path / "TestModel-confusion-matrix.png"
            mock_write_image.assert_called_once_with(expected_path)
        else:
            # If output_path is not provided, ensure the confusion matrix is shown
            mock_show.assert_called_once()


@pytest.mark.parametrize("output_path", [None, Path("/tmp/plots")])
def test_plot_most_common_tokens(mock_evaluation_result, output_path):
    plotter = Plotter(results=mock_evaluation_result, model_name="TestModel")

    with patch("pathlib.Path.mkdir") as mock_mkdir, patch("plotly.graph_objs.Figure.write_image") as mock_write_image, patch("plotly.graph_objs.Figure.show") as mock_show:
        plotter.plot_most_common_tokens(output_folder=output_path)

        if output_path:
            # If output_path is provided, ensure the plots are saved
            mock_mkdir.assert_called_once()
            expected_path = output_path / "TestModel-common-errors.png"
            mock_write_image.assert_called()  # Called for both false positives and false negatives
        else:
            # If output_path is not provided, ensure the plots are shown
            mock_show.assert_called()

def test_save_fig_to_file(mock_evaluation_result):
    plotter = Plotter(results=mock_evaluation_result, model_name="TestModel")
    mock_fig = MagicMock()

    with patch("pathlib.Path.mkdir") as mock_mkdir:
        plotter.save_fig_to_file(fig=mock_fig, output_folder=Path("/tmp"), file_name="test")
        mock_fig.write_image.assert_called_once_with(Path("/tmp", "TestModel-test.png"))
        mock_mkdir.assert_called_once()

def test_save_fig_to_correct_path(mock_evaluation_result):
    plotter = Plotter(results=mock_evaluation_result, model_name="TestModel")
    mock_fig = MagicMock()
    output_folder = Path("/tmp")
    file_name = "test"

    with patch("pathlib.Path.mkdir") as mock_mkdir:
        with patch("pathlib.Path.exists", return_value=True):
            plotter.save_fig_to_file(fig=mock_fig, output_folder=output_folder, file_name=file_name)
            expected_path = output_folder / f"TestModel-{file_name}.png"
            mock_fig.write_image.assert_called_once_with(expected_path)
            mock_mkdir.assert_called_once()