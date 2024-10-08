"""Unit tests for runners/skmlp_runner.py"""

import re
import threading
import time
import warnings
from unittest.mock import patch

import pytest
import sklearn.metrics as skmt

from mlrose_ky import SKMLPRunner
from mlrose_ky.neural import activation
from tests.globals import SEED


class TestSKMLPRunner:
    """Tests for SKMLPRunner."""

    @pytest.fixture
    def data(self):
        """Fixture to provide dummy training and test data."""
        x_train = [[0, 0], [1, 1], [1, 0], [0, 1]] * 10
        y_train = [0, 1, 1, 0] * 10
        x_test = [[0, 0], [1, 1], [1, 0], [0, 1]]
        y_test = [0, 1, 1, 0]
        return x_train, y_train, x_test, y_test

    @pytest.fixture
    def runner_kwargs(self, data):
        """Fixture to provide common kwargs for SKMLPRunner initialization."""
        x_train, y_train, x_test, y_test = data
        grid_search_parameters = {
            "max_iters": [10],
            "activation": [activation.relu, activation.sigmoid, activation.tanh, activation.identity],
            "learning_rate_init": [0.001],
            "max_attempts": [5],
            "solver": ["adam", "sgd", "lbfgs"],
        }

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "experiment_name": "test_experiment",
            "seed": SEED,
            "iteration_list": [1, 2],
            "grid_search_parameters": grid_search_parameters,
            "grid_search_scorer_method": skmt.balanced_accuracy_score,
            "early_stopping": True,
            "validation_fraction": 0.2,
            "max_attempts": 5,
            "n_jobs": 1,
            "cv": 2,
            "generate_curves": True,
        }

    @pytest.fixture
    def runner(self, runner_kwargs):
        """Fixture to initialize an SKMLPRunner instance."""
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            return SKMLPRunner(**runner_kwargs)

    def test_activation_mapping(self, runner_kwargs):
        """Test that all activation functions are correctly mapped."""
        runner = SKMLPRunner(**runner_kwargs)
        grid_params = runner.build_grid_search_parameters(runner.grid_search_parameters)
        expected_activations = ["relu", "logistic", "tanh", "identity"]
        assert set(grid_params["activation"]) == set(expected_activations)

    def test_invalid_activation_function(self, runner_kwargs):
        """Test that a ValueError is raised when a provided activation function is incompatible with MLPClassifier."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"MLPClassifier expects 'activation' to be a str among {'relu', 'logistic', 'tanh', 'identity'}, got {activation.softmax}."
            ),
        ):
            runner_kwargs["grid_search_parameters"]["activation"] = [
                *runner_kwargs["grid_search_parameters"]["activation"],
                activation.softmax,
            ]
            _ = SKMLPRunner(**runner_kwargs)

    def test_abort_during_run(self, runner_kwargs):
        """Test that aborting during run works and covers has_aborted code path."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            runner = SKMLPRunner(**runner_kwargs)

            # noinspection PyMissingOrEmptyDocstring
            def abort_runner():
                time.sleep(0.1)
                runner.abort()

            abort_thread = threading.Thread(target=abort_runner)
            abort_thread.start()
            runner.run()
            abort_thread.join()

        assert runner.has_aborted()

    def test_max_attempts_in_grid_search_parameters(self, runner_kwargs):
        """Test that max_attempts is correctly converted to n_iter_no_change."""
        runner = SKMLPRunner(**runner_kwargs)
        assert "n_iter_no_change" in runner.grid_search_parameters
        assert "max_attempts" not in runner.grid_search_parameters

    def test_solver_methods_called(self, runner_kwargs):
        """Test that the solver-specific methods are called."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            runner = SKMLPRunner(**runner_kwargs)
            runner.run()
