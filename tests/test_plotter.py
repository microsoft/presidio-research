import pytest
from pathlib import Path
import pandas as pd
from plotly.graph_objs import Figure
from unittest.mock import MagicMock, patch

from presidio_evaluator.evaluation import EvaluationResult, ModelError
from presidio_evaluator.evaluation.plotter import Plotter


@pytest.fixture
def mock_evaluation_result():
    result = MagicMock(spec=EvaluationResult)
    result.entity_recall_dict = {"PERSON": 0.8, "LOCATION": 0.7}
    result.entity_precision_dict = {"PERSON": 0.9, "LOCATION": 0.85}
    result.n_dict = {"PERSON": 100, "LOCATION": 50}
    result.pii_recall = 0.75
    result.pii_precision = 0.88
    result.pii_f = 0.8
    result.n = 150
    result.model_errors = []  # Empty list for test cases
    return result


@pytest.fixture
def mock_figure():
    fig = MagicMock(spec=Figure)
    fig.show = MagicMock()
    fig.write_image = MagicMock()
    return fig


def test_save_fig_to_file_creates_directory(mock_evaluation_result, tmp_path):
    # Setup
    output_dir = tmp_path / "test_output"
    plotter = Plotter(mock_evaluation_result)
    mock_fig = MagicMock(spec=Figure)

    # Execute
    plotter.save_fig_to_file(mock_fig, output_dir, "test-figure")

    # Assert
    assert output_dir.exists()
    mock_fig.write_image.assert_called_once_with(
        Path(output_dir, f"{plotter.model_name}-test-figure.png")
    )

def test_save_fig_to_file_different_formats(mock_evaluation_result, tmp_path):
    # Setup
    output_dir = tmp_path / "test_output"
    plotter = Plotter(mock_evaluation_result, save_as="svg")
    mock_fig = MagicMock(spec=Figure)

    # Execute
    plotter.save_fig_to_file(mock_fig, output_dir, "test-figure")

    # Assert
    mock_fig.write_image.assert_called_once_with(
        Path(output_dir, f"{plotter.model_name}-test-figure.svg")
    )

@patch("plotly.express.bar", return_value=MagicMock(spec=Figure))
def test_plot_scores_saves_multiple_figures(
    mock_px_bar, mock_evaluation_result, tmp_path
):
    # Setup
    output_dir = tmp_path / "test_output"
    plotter = Plotter(mock_evaluation_result)
    mock_fig = mock_px_bar.return_value

    # Execute
    plotter.plot_scores(output_dir)

    # Assert - should create 3 figures (beta, recall, precision)
    expected_paths = [
        str(Path(output_dir, f"{plotter.model_name}-scores-f_beta.png")),
        str(Path(output_dir, f"{plotter.model_name}-scores-recall.png")),
        str(Path(output_dir, f"{plotter.model_name}-scores-precision.png")),
    ]
    assert mock_fig.write_image.call_count == 3
    actual_paths = [str(call[0][0]) for call in mock_fig.write_image.call_args_list]
    assert actual_paths == expected_paths

@patch("presidio_evaluator.evaluation.model_error.ModelError.get_fps_dataframe")
@patch("presidio_evaluator.evaluation.model_error.ModelError.get_fns_dataframe")
@patch("plotly.express.histogram", return_value=MagicMock(spec=Figure))
def test_plot_most_common_tokens_saves_figures(
    mock_px_histogram,
    mock_get_fns,
    mock_get_fps,
    mock_evaluation_result,
    tmp_path,
):
    # Setup
    output_dir = tmp_path / "test_output"
    plotter = Plotter(mock_evaluation_result)

    # Mock dataframes with some test data
    test_data = {
        "token": ["test1", "test2"],
        "prediction": ["PERSON", "LOCATION"],
        "annotation": ["PERSON", "LOCATION"],
    }
    mock_get_fps.return_value = pd.DataFrame(test_data)
    mock_get_fns.return_value = pd.DataFrame(test_data)
    mock_fig = mock_px_histogram.return_value

    # Execute
    plotter.plot_most_common_tokens(output_dir)

    # Assert - should create 2 figures (false positives and false negatives)
    expected_paths = [
        str(Path(output_dir, f"{plotter.model_name}-common-errors-fn.png")),
        str(Path(output_dir, f"{plotter.model_name}-common-errors-fp.png")),
    ]
    assert mock_fig.write_image.call_count == 2
    actual_paths = [str(call[0][0]) for call in mock_fig.write_image.call_args_list]
    assert actual_paths == expected_paths

@patch("plotly.express.imshow", return_value=MagicMock(spec=Figure))
def test_plot_confusion_matrix_saves_figure(
    mock_px_imshow, mock_evaluation_result, tmp_path
):
    # Setup
    output_dir = tmp_path / "test_output"
    plotter = Plotter(mock_evaluation_result)
    mock_fig = mock_px_imshow.return_value
    entities = ["PERSON", "LOCATION"]
    conf_matrix = [[10, 2], [1, 8]]

    # Execute
    plotter.plot_confusion_matrix(entities, conf_matrix, output_dir)

    # Assert
    expected_path = str(
        Path(output_dir, f"{plotter.model_name}-confusion-matrix.png")
    )
    mock_fig.write_image.assert_called_once_with(
        Path(output_dir, f"{plotter.model_name}-confusion-matrix.png")
    )
    assert str(mock_fig.write_image.call_args[0][0]) == expected_path

def test_special_chars_in_model_name(mock_evaluation_result, tmp_path):
    # Setup
    output_dir = tmp_path / "test_output"
    plotter = Plotter(mock_evaluation_result, model_name="model/with/slashes")
    mock_fig = MagicMock(spec=Figure)

    # Execute
    plotter.save_fig_to_file(mock_fig, output_dir, "test-figure")

    # Assert - slashes should be replaced with hyphens
    expected_path = str(Path(output_dir, "model-with-slashes-test-figure.png"))
    mock_fig.write_image.assert_called_once_with(
        Path(output_dir, "model-with-slashes-test-figure.png")
    )
    assert str(mock_fig.write_image.call_args[0][0]) == expected_path

def test_plot_scores_without_output_folder(mock_evaluation_result):
    # Setup
    plotter = Plotter(mock_evaluation_result)
    with patch(
        "plotly.express.bar", return_value=MagicMock(spec=Figure)
    ) as mock_px_bar:
        mock_fig = mock_px_bar.return_value

        # Execute
        plotter.plot_scores(None)

        # Assert - figures should only be shown, not saved
        mock_fig.write_image.assert_not_called()
        assert mock_fig.show.call_count == 3

def test_plot_most_common_tokens_empty_data(mock_evaluation_result, tmp_path):
    # Setup
    output_dir = tmp_path / "test_output"
    plotter = Plotter(mock_evaluation_result)

    with patch(
        "presidio_evaluator.evaluation.model_error.ModelError.get_fps_dataframe"
    ) as mock_get_fps:
        with patch(
            "presidio_evaluator.evaluation.model_error.ModelError.get_fns_dataframe"
        ) as mock_get_fns:
            # Mock empty dataframes
            mock_get_fps.return_value = None
            mock_get_fns.return_value = None

            # Execute
            plotter.plot_most_common_tokens(output_dir)

            # Assert - no figures should be saved when there's no data
            # The function should complete without errors
            assert True
