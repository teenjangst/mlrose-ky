"""Unit tests for algorithms/sa.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky import DiscreteOpt, ContinuousOpt, OneMax
from mlrose_ky.algorithms import simulated_annealing
from tests.globals import SEED


class TestSimulatedAnnealing:
    """Unit tests for simulated_annealing."""

    def test_simulated_annealing_invalid_max_attempts(self):
        """Test that simulated_annealing raises ValueError when max_attempts is invalid."""
        problem = DiscreteOpt(5, OneMax())
        max_attempts = -1
        with pytest.raises(ValueError, match=f"max_attempts must be a positive integer. Got {max_attempts}"):
            simulated_annealing(problem, max_attempts=max_attempts, random_state=SEED)

    def test_simulated_annealing_invalid_max_iters(self):
        """Test that simulated_annealing raises ValueError when max_iters is invalid."""
        problem = DiscreteOpt(5, OneMax())
        max_iters = -1
        with pytest.raises(ValueError, match=f"max_iters must be a positive integer or np.inf. Got {max_iters}"):
            simulated_annealing(problem, max_iters=max_iters, random_state=SEED)

    def test_simulated_annealing_invalid_init_state_length(self):
        """Test that simulated_annealing raises ValueError when init_state length is invalid."""
        problem = DiscreteOpt(5, OneMax())
        init_state = np.zeros(4)  # Incorrect length
        with pytest.raises(
            ValueError, match=f"init_state must have the same length as the problem. Expected {problem.get_length()}, got {len(init_state)}"
        ):
            simulated_annealing(problem, init_state=init_state, random_state=SEED)

    def test_simulated_annealing_invalid_callback_user_info_type(self):
        """Test that simulated_annealing raises TypeError when callback_user_info is not a dict."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(TypeError, match=f"callback_user_info must be a dict. Got str"):
            # noinspection PyTypeChecker
            simulated_annealing(problem, callback_user_info="User data should be a dict")

    def test_simulated_annealing_discrete_max(self):
        """Test simulated_annealing function for a discrete maximization problem"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = simulated_annealing(problem, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_simulated_annealing_continuous_max(self):
        """Test simulated_annealing function for a continuous maximization problem"""
        problem = ContinuousOpt(5, OneMax())
        best_state, best_fitness, _ = simulated_annealing(problem, max_attempts=20, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_simulated_annealing_discrete_min(self):
        """Test simulated_annealing function for a discrete minimization problem"""
        problem = DiscreteOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = simulated_annealing(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_simulated_annealing_continuous_min(self):
        """Test simulated_annealing function for a continuous minimization problem"""
        problem = ContinuousOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = simulated_annealing(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_simulated_annealing_max_iters(self):
        """Test simulated_annealing function with low max_iters"""
        problem = DiscreteOpt(5, OneMax())
        x = np.zeros(5)
        best_state, best_fitness, _ = simulated_annealing(problem, max_attempts=1, max_iters=1, init_state=x, random_state=SEED)
        assert best_fitness == 1

    def test_simulated_annealing_with_callback(self):
        """Test simulated_annealing with a state_fitness_callback."""
        problem = DiscreteOpt(5, OneMax())

        # Define a callback function
        def callback_function(iteration, attempt, done, state, fitness, fitness_evaluations, curve, user_data):
            # Record the iteration number
            user_data["iterations"].append(iteration)
            return True  # Continue iterating

        callback_data = {"iterations": []}
        simulated_annealing(problem, random_state=SEED, state_fitness_callback=callback_function, callback_user_info=callback_data)

        # Assert that the callback was called
        assert len(callback_data["iterations"]) > 0

    def test_simulated_annealing_zero_temperature(self):
        """Test simulated_annealing where temperature becomes zero."""
        problem = DiscreteOpt(5, OneMax())

        # noinspection PyMissingOrEmptyDocstring
        class ZeroTempSchedule:
            @staticmethod
            def evaluate(t):
                if t < 3:
                    return 1.0
                else:
                    return 0.0

        schedule = ZeroTempSchedule()

        best_state, best_fitness, fitness_curve = simulated_annealing(problem, schedule=schedule, random_state=SEED, curve=True)

        # Since temperature becomes zero, the loop should terminate early
        # Assert that the number of iterations is less than max_iters
        assert len(fitness_curve) == 3  # Should have 3 iterations before temp reaches zero

    def test_simulated_annealing_callback_stop(self):
        """Test simulated_annealing where the callback stops the iteration."""
        problem = DiscreteOpt(5, OneMax())

        # Define a callback function that stops after 5 iterations
        # noinspection PyMissingOrEmptyDocstring
        def callback_function(iteration, attempt, done, state, fitness, fitness_evaluations, curve, user_data):
            if iteration >= 5:
                return False  # Stop iterating
            else:
                return True  # Continue iterating

        simulated_annealing(problem, random_state=SEED, state_fitness_callback=callback_function)

        # Assert that the algorithm stopped at iteration 5
        assert problem.current_iteration == 5

    def test_simulated_annealing_max_attempts_reached(self):
        """Test simulated_annealing where max_attempts is reached."""
        problem = DiscreteOpt(5, OneMax())

        # Create an initial state that is already the maximum
        init_state = np.ones(5)

        # noinspection PyMissingOrEmptyDocstring
        def callback_function(iteration, attempt, done, state, fitness, fitness_evaluations, curve, user_data):
            # Record the attempts
            user_data["attempts"].append(attempt)
            return True

        max_attempts = 3
        user_data = {"attempts": []}

        # Since the initial state is already optimal, no better state will be found,
        # so attempts will increase until max_attempts is reached.
        simulated_annealing(
            problem,
            init_state=init_state,
            max_attempts=max_attempts,
            random_state=SEED,
            state_fitness_callback=callback_function,
            callback_user_info=user_data,
        )

        # Check that max_attempts was reached
        assert max(user_data["attempts"]) == max_attempts
