"""Unit tests for algorithms/rhc.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky import DiscreteOpt, ContinuousOpt, OneMax
from mlrose_ky.algorithms import random_hill_climb
from tests.globals import SEED


class TestRandomHillClimb:
    """Unit tests for random_hill_climb."""

    def test_random_hill_climb_invalid_max_attempts(self):
        """Test that random_hill_climb raises ValueError when max_attempts is invalid."""
        problem = DiscreteOpt(5, OneMax())
        max_attempts = -1
        with pytest.raises(ValueError, match=f"max_attempts must be a positive integer. Got {max_attempts}"):
            random_hill_climb(problem, max_attempts=max_attempts, random_state=SEED)

    def test_random_hill_climb_invalid_max_iters(self):
        """Test that random_hill_climb raises ValueError when max_iters is invalid."""
        problem = DiscreteOpt(5, OneMax())
        max_iters = -1
        with pytest.raises(ValueError, match=f"max_iters must be a positive integer or np.inf. Got {max_iters}"):
            random_hill_climb(problem, max_iters=max_iters, random_state=SEED)

    def test_random_hill_climb_invalid_restarts(self):
        """Test that random_hill_climb raises ValueError when restarts is invalid."""
        problem = DiscreteOpt(5, OneMax())
        restarts = -1
        with pytest.raises(ValueError, match=f"restarts must be a positive integer. Got {restarts}"):
            random_hill_climb(problem, restarts=restarts, random_state=SEED)

    def test_random_hill_climb_invalid_init_state_length(self):
        """Test that random_hill_climb raises ValueError when init_state length is invalid."""
        problem = DiscreteOpt(5, OneMax())
        init_state = np.zeros(4)  # Incorrect length
        with pytest.raises(
            ValueError, match=f"init_state must have the same length as the problem. Expected {problem.get_length()}, got {len(init_state)}"
        ):
            random_hill_climb(problem, init_state=init_state, random_state=SEED)

    def test_random_hill_climb_invalid_callback_user_info_type(self):
        """Test that random_hill_climb raises TypeError when callback_user_info is not a dict."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(TypeError, match=f"callback_user_info must be a dict. Got str"):
            # noinspection PyTypeChecker
            random_hill_climb(problem, callback_user_info="User data should be a dict")

    def test_random_hill_climb_discrete_max(self):
        """Test random_hill_climb function for a discrete maximization problem"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = random_hill_climb(problem, restarts=10, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_random_hill_climb_continuous_max(self):
        """Test random_hill_climb function for a continuous maximization problem"""
        problem = ContinuousOpt(5, OneMax())
        best_state, best_fitness, _ = random_hill_climb(problem, restarts=10, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_random_hill_climb_discrete_min(self):
        """Test random_hill_climb function for a discrete minimization problem"""
        problem = DiscreteOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = random_hill_climb(problem, restarts=10, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_random_hill_climb_continuous_min(self):
        """Test random_hill_climb function for a continuous minimization problem"""
        problem = ContinuousOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = random_hill_climb(problem, restarts=10, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_random_hill_climb_max_iters(self):
        """Test random_hill_climb function with low max_iters"""
        problem = DiscreteOpt(5, OneMax())
        x = np.zeros(5)
        best_state, best_fitness, _ = random_hill_climb(problem, max_attempts=1, max_iters=1, init_state=x, random_state=SEED)
        assert best_fitness == 1

    def test_random_hill_climb_with_callback(self):
        """Test random_hill_climb with a state_fitness_callback."""
        problem = DiscreteOpt(5, OneMax())

        # noinspection PyMissingOrEmptyDocstring
        def callback_function(iteration, attempt, done, state, fitness, fitness_evaluations, curve, user_data):
            user_data["iterations"].append(iteration)
            return True  # Continue iterating

        callback_info = {"iterations": []}
        random_hill_climb(problem, random_state=SEED, state_fitness_callback=callback_function, callback_user_info=callback_info)

        assert len(callback_info["iterations"]) > 0

    def test_random_hill_climb_callback_stop(self):
        """Test random_hill_climb where the callback stops the iteration."""
        problem = DiscreteOpt(5, OneMax())

        # noinspection PyMissingOrEmptyDocstring
        def callback_function(iteration, attempt, done, state, fitness, fitness_evaluations, curve, user_data):
            if iteration >= 5:
                return False  # Stop iterating
            else:
                return True  # Continue iterating

        random_hill_climb(problem, random_state=SEED, state_fitness_callback=callback_function)

        assert problem.current_iteration == 5

    def test_random_hill_climb_max_attempts_reached(self):
        """Test random_hill_climb where max_attempts is reached."""
        problem = DiscreteOpt(5, OneMax())

        init_state = np.ones(5)

        # noinspection PyMissingOrEmptyDocstring
        def callback_function(iteration, attempt, done, state, fitness, fitness_evaluations, curve, user_data):
            user_data["attempts_list"].append(attempt)
            return True

        callback_info = {"attempts_list": []}
        random_hill_climb(
            problem,
            init_state=init_state,
            max_attempts=3,
            random_state=SEED,
            state_fitness_callback=callback_function,
            callback_user_info=callback_info,
        )

        assert max(callback_info["attempts_list"]) == 3

    def test_random_hill_climb_max_iters_reached(self):
        """Test random_hill_climb where max_iters is reached."""
        problem = DiscreteOpt(5, OneMax())

        best_state, best_fitness, fitness_curve = random_hill_climb(problem, max_iters=2, random_state=SEED, curve=True)

        assert len(fitness_curve) == 2
        assert problem.current_iteration == 2

    def test_random_hill_climb_problem_can_stop(self):
        """Test random_hill_climb where problem.can_stop() becomes True."""

        class CustomProblem(DiscreteOpt):
            def can_stop(self):
                return self.current_iteration >= 3

        problem = CustomProblem(5, OneMax())
        best_state, best_fitness, fitness_curve = random_hill_climb(problem, random_state=SEED, curve=True)

        assert len(fitness_curve) == 3
        assert problem.current_iteration == 3

    def test_random_hill_climb_zero_iterations(self):
        """Test random_hill_climb where max_iters is zero."""
        problem = DiscreteOpt(5, OneMax())

        best_state, best_fitness, fitness_curve = random_hill_climb(problem, max_iters=0, random_state=SEED, curve=True)

        assert len(fitness_curve) == 0
        assert problem.current_iteration == 0

    def test_random_hill_climb_callback_stop_at_iteration_0(self):
        """Test random_hill_climb where the callback stops the iteration at 0."""
        problem = DiscreteOpt(5, OneMax())

        # noinspection PyMissingOrEmptyDocstring
        def callback_function(iteration, attempt, done, state, fitness, fitness_evaluations, curve, user_data):
            return False  # Stop iterating immediately

        best_state, best_fitness, fitness_curve = random_hill_climb(
            problem, random_state=SEED, state_fitness_callback=callback_function, curve=True
        )

        assert problem.current_iteration == 0
        assert len(fitness_curve) == 0
        assert best_state is not None
        assert best_fitness == -np.inf
