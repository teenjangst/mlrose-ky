"""Unit tests for runners/_runner_base.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import pickle as pk
import signal
from unittest.mock import patch, Mock, mock_open

import numpy as np
import pandas as pd
import pytest

# noinspection PyProtectedMember
from mlrose_ky.runners._runner_base import _RunnerBase
from tests.globals import SEED


class TestRunnerBase:
    # noinspection PyUnresolvedReferences
    @pytest.fixture(autouse=True)
    def reset_runner_base_class_vars(self):
        """Fixture to reset _RunnerBase attributes before each test."""
        _RunnerBase._RunnerBase__abort.value = False
        _RunnerBase._RunnerBase__spawn_count.value = 0
        _RunnerBase._RunnerBase__replay.value = False
        _RunnerBase._RunnerBase__original_sigint_handler = None
        _RunnerBase._RunnerBase__sigint_params = None
        yield

    @pytest.fixture
    def _test_runner_fixture(self):
        """Fixture to create a TestRunner instance for testing."""

        # noinspection PyMissingOrEmptyDocstring
        class TestRunner(_RunnerBase):
            def run(self):
                pass

        def _create_runner(**kwargs):
            default_kwargs = {
                "problem": None,
                "experiment_name": "test_experiment",
                "seed": 1,
                "iteration_list": [1, 2, 3],
                "output_directory": "test_output",
                "override_ctrl_c_handler": False,
            }
            # Update default_kwargs with any provided kwargs
            default_kwargs.update(kwargs)
            return TestRunner(**default_kwargs)

        return _create_runner

    def test_increment_spawn_count(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture()
            initial_count = runner._get_spawn_count()
            runner._increment_spawn_count()

            assert runner._get_spawn_count() == initial_count + 1

    def test_decrement_spawn_count(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture()
            runner._increment_spawn_count()
            initial_count = runner._get_spawn_count()
            runner._decrement_spawn_count()

            assert runner._get_spawn_count() == initial_count - 1

    def test_get_spawn_count(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture()
            initial_spawn_count = runner._get_spawn_count()
            runner._increment_spawn_count()
            incremented_spawn_count = runner._get_spawn_count()
            assert incremented_spawn_count == initial_spawn_count + 1

            runner._decrement_spawn_count()
            decremented_spawn_count = runner._get_spawn_count()
            assert decremented_spawn_count == initial_spawn_count

    def test_abort_sets_abort_flag(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture()
            runner.abort()

            assert runner.has_aborted() is True

    def test_has_aborted_after_abort_called(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture(seed=SEED, iteration_list=[0])
            runner.abort()

            assert runner.has_aborted() is True

    def test_set_replay_mode(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture()
            assert not runner.replay_mode()

            runner.set_replay_mode()
            assert runner.replay_mode()

            runner.set_replay_mode(False)
            assert not runner.replay_mode()

    def test_replay_mode(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture(replay=True)
            assert runner.replay_mode() is True

            runner.set_replay_mode(False)
            assert runner.replay_mode() is False

    def test_setup_method(self, _test_runner_fixture):
        with patch("os.makedirs") as mock_makedirs, patch("os.path.exists", return_value=False):
            runner = _test_runner_fixture(problem="dummy_problem", seed=SEED, iteration_list=[0, 1, 2], output_directory="test_output")
            runner._setup()

            assert runner._raw_run_stats == []
            assert runner._fitness_curves == []
            assert runner._curve_base == 0
            assert runner._iteration_times == []
            assert runner._copy_zero_curve_fitness_from_first == runner._copy_zero_curve_fitness_from_first_original
            assert runner._current_logged_algorithm_args == {}
            mock_makedirs.assert_called_once_with("test_output")

    def test_tear_down_restores_original_sigint_handler(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            original_handler = signal.getsignal(signal.SIGINT)
            runner = _test_runner_fixture()
            runner.__original_sigint_handler = original_handler  # Ensure original handler is set
            runner._tear_down()
            restored_handler = signal.getsignal(signal.SIGINT)

            assert restored_handler == original_handler

    def test_log_current_argument(self, _test_runner_fixture):
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            runner = _test_runner_fixture(seed=SEED, iteration_list=[0, 1, 2])
            arg_name = "test_arg"
            arg_value = "test_value"
            runner._log_current_argument(arg_name, arg_value)

            assert runner._current_logged_algorithm_args[arg_name] == arg_value

    def test_sanitize_value_with_list(self, _test_runner_fixture):
        """Test that _sanitize_value correctly handles list inputs."""
        runner = _test_runner_fixture()
        value = [1, 2, 3]
        sanitized = runner._sanitize_value(value)
        assert sanitized == "[1, 2, 3]"

    def test_sanitize_value_with_tuple(self, _test_runner_fixture):
        """Test that _sanitize_value correctly handles tuple inputs."""
        runner = _test_runner_fixture()
        value = (1, 2, 3)
        sanitized = runner._sanitize_value(value)
        assert sanitized == "(1, 2, 3)"

    def test_sanitize_value_with_ndarray(self, _test_runner_fixture):
        """Test that _sanitize_value correctly handles numpy.ndarray inputs."""
        runner = _test_runner_fixture()
        value = np.array([1, 2, 3])
        sanitized = runner._sanitize_value(value)
        assert sanitized == "[1, 2, 3]"

    def test_sanitize_value_with_other(self, _test_runner_fixture):
        """Test that _sanitize_value correctly handles other types."""
        runner = _test_runner_fixture()
        mock_object = Mock()
        mock_object.__name__ = "MockObject"
        sanitized = runner._sanitize_value(mock_object)
        assert sanitized == "MockObject"

    def test_run_method_is_abstract(self, _test_runner_fixture):
        """Test that _RunnerBase cannot be instantiated without implementing run method."""
        with pytest.raises(TypeError):
            _RunnerBase(problem=None, experiment_name="test_experiment", seed=1, iteration_list=[1, 2, 3], output_directory="test_output")

    def test_load_pickles_success(self, _test_runner_fixture):
        """Test that _load_pickles loads pickles correctly when files exist and are valid."""
        runner = _test_runner_fixture()
        runner._output_directory = "test_output"

        # Mock _get_pickle_filename_root
        with (
            patch.object(runner, "_get_pickle_filename_root", side_effect=lambda name: f"test_output/{name}"),
            patch("os.path.exists", side_effect=lambda filename: filename in ["test_output/curves_df.p", "test_output/run_stats_df.p"]),
            patch("pickle.load", side_effect=lambda f: pd.DataFrame({"Iteration": [1, 2], "Fitness": [0.8, 0.85]})),
            patch("builtins.open", mock_open()),
        ):
            result = runner._load_pickles()

            # Check that curves_df and run_stats_df are loaded
            assert result is True
            assert runner.curves_df is not None
            assert runner.run_stats_df is not None

    def test_load_pickles_with_pickling_exception(self, _test_runner_fixture):
        """Test that _load_pickles handles pickle loading exceptions gracefully."""
        runner = _test_runner_fixture()
        runner._output_directory = "test_output"

        # Mock _get_pickle_filename_root
        with (
            patch.object(runner, "_get_pickle_filename_root", side_effect=lambda name: f"test_output/{name}"),
            patch("os.path.exists", side_effect=lambda filename: filename in ["test_output/curves_df.p", "test_output/run_stats_df.p"]),
            patch("pickle.load", side_effect=pk.PickleError),
            patch("builtins.open", mock_open()),
        ):
            result = runner._load_pickles()

            # Both curves_df and run_stats_df should be None because of exception
            assert result is False
            assert runner.curves_df is None
            assert runner.run_stats_df is None

    def test_invoke_algorithm_with_replay_mode_enabled_and_pickles_loaded(self, _test_runner_fixture):
        """Test that _invoke_algorithm returns None, None, None when replay mode is enabled and pickles are loaded."""
        mock_problem = Mock()
        runner = _test_runner_fixture(problem=mock_problem)
        runner.set_replay_mode()

        def mock_algorithm(problem, max_attempts, curve, random_state, state_fitness_callback, callback_user_info):
            return {"result": "success"}

        # Mock _load_pickles to return True
        with (
            patch.object(runner, "_load_pickles", return_value=True),
            patch.object(runner, "_print_banner"),
            patch.object(runner, "_start_run_timing"),
            patch.object(runner.problem, "reset"),
        ):
            result = runner._invoke_algorithm(
                algorithm=mock_algorithm, problem=runner.problem, max_attempts=100, curve=True, callback_user_info={}
            )

            # Since _load_pickles returns True, it should return (None, None, None) or just None
            assert result == (None, None, None) or result is None

    def test_invoke_algorithm_executes_algorithm_when_replay_mode_disabled(self, _test_runner_fixture):
        """Test that _invoke_algorithm executes the algorithm when replay mode is disabled."""
        mock_problem = Mock()
        runner = _test_runner_fixture(problem=mock_problem)
        runner.set_replay_mode(False)

        # Mock _load_pickles to return False and other dependencies
        with (
            patch.object(runner, "_load_pickles", return_value=False),
            patch.object(runner, "_print_banner"),
            patch.object(runner, "_start_run_timing"),
            patch.object(runner.problem, "reset"),
        ):
            # Mock the algorithm function
            mock_algorithm_func = Mock(return_value={"result": "success"})

            # Invoke the algorithm
            result = runner._invoke_algorithm(
                algorithm=mock_algorithm_func, problem=runner.problem, max_attempts=100, curve=True, callback_user_info={}
            )

            # Check that the algorithm was called with correct arguments
            mock_algorithm_func.assert_called_once_with(
                problem=runner.problem,
                max_attempts=100,
                curve=True,
                random_state=runner.seed,
                state_fitness_callback=runner._save_state,
                callback_user_info={},
            )

            # Check the result
            assert result == {"result": "success"}

    def test_save_state_returns_true_when_iteration_not_in_list_and_not_done(self, _test_runner_fixture):
        """Test that _save_state returns True when iteration is not in iteration_list and not done."""
        runner = _test_runner_fixture(iteration_list=[1, 2, 3])
        runner._setup()  # Initialize _run_start_time and other setups
        runner._start_run_timing()
        with (
            patch.object(runner, "_sanitize_value"),
            patch("logging.debug"),
            patch.object(runner, "_create_curve_stat"),
            patch.object(runner, "_create_and_save_run_data_frames"),
        ):
            # Mock perf_counter to return a fixed time
            with patch("time.perf_counter", side_effect=[100.0, 100.5]):
                # Call _save_state with iteration=4 (not in [1,2,3]) and done=False
                result = runner._save_state(iteration=4, state="state4", fitness=0.9, user_data={})

                # Check that the result is True
                assert result is True

    def test_save_state_sets_curve_when_generate_curves_true_and_iteration_zero_and_curve_none(self, _test_runner_fixture):
        """Test that _save_state sets curve and _first_curve_synthesized when generate_curves is True, iteration is 0, and curve is None."""
        runner = _test_runner_fixture(iteration_list=[0, 1, 2], generate_curves=True)
        runner._setup()  # Initialize _run_start_time and other setups
        runner._start_run_timing()

        with (
            patch.object(runner, "_sanitize_value"),
            patch("logging.debug"),
            patch.object(runner, "_create_curve_stat"),
            patch.object(runner, "_create_and_save_run_data_frames"),
        ):
            # Mock perf_counter to return a fixed time
            with patch("time.perf_counter", side_effect=[100.0, 100.0]):
                # Call _save_state with iteration=0 and curve=None
                result = runner._save_state(iteration=0, state="state0", fitness=0.95, user_data={})

                # Check that the result is True (should continue)
                assert result is True

                # Check that _first_curve_synthesized is set to True
                assert runner._first_curve_synthesized is True

    def test_create_curve_stat_with_dict_curve_value(self, _test_runner_fixture):
        """Test that _create_curve_stat correctly updates with dict curve_value."""
        runner = _test_runner_fixture()

        curve_value = {"Fitness": 0.95, "FEvals": 10}
        curve_data = {"param1": "value1"}
        iteration = 1

        # Call _create_curve_stat
        curve_stat = runner._create_curve_stat(iteration=iteration, curve_value=curve_value, curve_data=curve_data)

        assert curve_stat["Iteration"] == 1
        assert curve_stat["Time"] is None
        assert curve_stat["Fitness"] == 0.95
        assert curve_stat["FEvals"] == 10
        assert curve_stat["param1"] == "value1"

    def test_dump_pickle_to_disk_handles_no_output_directory(self, _test_runner_fixture):
        """Test that _dump_pickle_to_disk returns None when output_directory is None."""
        runner = _test_runner_fixture()
        runner._output_directory = None

        # Call _dump_pickle_to_disk
        filename_root = runner._dump_pickle_to_disk(object_to_pickle={"data": 123}, name="run_stats_df", final_save=True)

        # Check that it returns None
        assert filename_root is None

    def test_dump_pickle_to_disk_saves_correctly(self, _test_runner_fixture):
        """Test that _dump_pickle_to_disk pickles the object and logs when final_save is True."""
        runner = _test_runner_fixture()
        runner._output_directory = "test_output"

        # Mock _get_pickle_filename_root and open
        with (
            patch.object(runner, "_get_pickle_filename_root", return_value="test_output/run_stats_df"),
            patch("pickle.dump") as mock_pickle_dump,
            patch("logging.info") as mock_logging,
            patch("builtins.open", mock_open()) as mock_file,  # Capture the mock_open as mock_file
        ):
            # Call _dump_pickle_to_disk
            filename_root = runner._dump_pickle_to_disk(object_to_pickle={"data": 123}, name="run_stats_df", final_save=True)

            # Check that pickle.dump was called correctly with the same mock file handle
            # noinspection PyArgumentList
            mock_pickle_dump.assert_called_once_with({"data": 123}, mock_file())

            # Check that logging was called for final save
            mock_logging.assert_called_once_with("Saved: [test_output/run_stats_df.p]")

            # Check the returned filename_root
            assert filename_root == "test_output/run_stats_df"

    def test_invoke_algorithm_updates_current_logged_algorithm_args_with_extra_args(self, _test_runner_fixture):
        """Test that _invoke_algorithm updates _current_logged_algorithm_args with _extra_args."""
        mock_problem = Mock()
        runner = _test_runner_fixture(problem=mock_problem)
        runner._extra_args = {"extra_param1": "value1", "extra_param2": "value2"}

        # Define a mock algorithm that accepts the extra parameters
        # noinspection PyMissingOrEmptyDocstring
        def mock_algorithm(
            problem, max_attempts, curve, random_state, state_fitness_callback, callback_user_info, extra_param1=None, extra_param2=None
        ):
            return {"result": "success"}

        # Mock dependencies
        with (
            patch.object(runner, "_load_pickles", return_value=False),
            patch.object(runner, "_print_banner"),
            patch.object(runner, "_start_run_timing"),
            patch.object(runner.problem, "reset"),
        ):
            # Invoke the algorithm
            runner._invoke_algorithm(algorithm=mock_algorithm, problem=runner.problem, max_attempts=100, curve=True, callback_user_info={})

            # Check that _current_logged_algorithm_args includes _extra_args
            assert runner._current_logged_algorithm_args["extra_param1"] == "value1"
            assert runner._current_logged_algorithm_args["extra_param2"] == "value2"

    def test_runner_base_abstract_method(self):
        """Test that calling abstract method raises NotImplementedError."""

        class TestRunner(_RunnerBase):

            def run(self):
                super().run()

        runner = TestRunner(problem=None, experiment_name="test", seed=1, iteration_list=[1, 2, 3])

        with pytest.raises(NotImplementedError, match="Subclasses must implement run method"):
            runner.run()
