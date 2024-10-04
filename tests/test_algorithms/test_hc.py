"""Unit tests for algorithms/hc.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky import DiscreteOpt, ContinuousOpt, OneMax
from mlrose_ky.algorithms import hill_climb
from tests.globals import SEED


class TestHillClimb:
    """Unit tests for hill_climb."""

    def test_hill_climb_discrete_max(self):
        """Test hill_climb function for a discrete maximization problem"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = hill_climb(problem, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_hill_climb_continuous_max(self):
        """Test hill_climb function for a continuous maximization problem"""
        problem = ContinuousOpt(5, OneMax())
        best_state, best_fitness, _ = hill_climb(problem, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_hill_climb_discrete_min(self):
        """Test hill_climb function for a discrete minimization problem"""
        problem = DiscreteOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = hill_climb(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_hill_climb_continuous_min(self):
        """Test hill_climb function for a continuous minimization problem"""
        problem = ContinuousOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = hill_climb(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_hill_climb_max_iters(self):
        """Test hill_climb function with max_iters less than infinite"""
        problem = DiscreteOpt(5, OneMax())
        x = np.zeros(5)
        best_state, best_fitness, _ = hill_climb(problem, max_iters=1, init_state=x, random_state=SEED)
        assert best_fitness == 1

    def test_hill_climb_callback_stop_at_iteration_0(self):
        """Test hill_climb where the callback stops the iteration at 0."""
        problem = DiscreteOpt(5, OneMax())
        init_state = np.zeros(5)

        # Define a callback function that stops after 0 iterations
        # noinspection PyMissingOrEmptyDocstring
        def callback_function(iteration, attempt, done, state, fitness, curve, user_data):
            return False

        # Call hill_climb with the callback function
        best_state, best_fitness, fitness_curve = hill_climb(
            problem, init_state=init_state, random_state=SEED, state_fitness_callback=callback_function
        )

        # Assert that no iterations were performed (i.e., algorithm stopped immediately)
        assert problem.current_iteration == 0
        assert np.array_equal(best_state, init_state)  # Since no iterations happened, the best state is the initial state
        assert best_fitness == -np.inf  # Since no iterations happened, the best fitness should remain the initial value
        assert fitness_curve is None  # Since curve is False by default

    def test_hill_climb_invalid_max_iters(self):
        """Test that hill_climb raises ValueError when max_iters is invalid."""
        problem = DiscreteOpt(5, OneMax())
        max_iters = -1
        with pytest.raises(ValueError, match=f"max_iters must be a positive integer or np.inf. Got {max_iters}"):
            hill_climb(problem, max_iters=max_iters, random_state=SEED)

    def test_hill_climb_invalid_init_state_length(self):
        """Test that hill_climb raises ValueError when init_state length is invalid."""
        problem = DiscreteOpt(5, OneMax())
        init_state = np.zeros(4)  # Incorrect length
        with pytest.raises(
            ValueError, match=f"init_state must have the same length as the problem. Expected {problem.get_length()}, got {len(init_state)}"
        ):
            hill_climb(problem, init_state=init_state, random_state=SEED)

    def test_hill_climb_with_callback(self):
        """Test hill_climb with a state_fitness_callback."""
        problem = DiscreteOpt(5, OneMax())

        # noinspection PyMissingOrEmptyDocstring
        def callback_function(iteration, attempt, done, state, fitness, curve, user_data):
            user_data["iterations"].append(iteration)
            return True  # Continue iterating

        callback_data = {"iterations": []}
        hill_climb(problem, random_state=SEED, state_fitness_callback=callback_function, callback_user_info=callback_data)

        # Assert that the callback was called
        assert len(callback_data["iterations"]) > 0

    def test_hill_climb_with_curve(self):
        """Test hill_climb function with curve=True."""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, fitness_curve = hill_climb(problem, curve=True, random_state=SEED)
        assert fitness_curve is not None
        assert len(fitness_curve) > 0

    def test_hill_climb_callback_stop(self):
        """Test hill_climb where the callback stops the iteration."""
        problem = DiscreteOpt(5, OneMax())
        init_state = np.zeros(5)

        # noinspection PyMissingOrEmptyDocstring
        def callback_function(iteration, attempt, done, state, fitness, curve, user_data):
            if iteration >= 5:
                return False  # Stop iterating
            else:
                return True  # Continue iterating

        # Since the initial state is very sub-optimal, a better state will certainly be found, so iterations will increase
        hill_climb(problem, init_state=init_state, random_state=SEED, state_fitness_callback=callback_function)

        # Assert that the algorithm stopped at iteration 5
        assert problem.current_iteration == 5

    def test_hill_climb_can_stop(self):
        """Test hill_climb function when can_stop is triggered."""

        early_stop_iter = 2

        # noinspection PyMissingOrEmptyDocstring
        class MockProblem:
            def __init__(self):
                # Initialize the problem state as an 8-element array of zeros to simulate the initial state
                self.current_iteration = 0
                self.state = np.zeros(8)
                self.fitness_evaluations = 0
                self.maximize = 1

            def reset(self):
                # Reset the state to the initial configuration
                self.state = np.zeros(8)

            def set_state(self, state):
                # Set the current state
                self.state = state

            def get_state(self):
                # Return the current state
                return self.state

            def get_fitness(self):
                # Return the current iteration count as the fitness to simulate improving fitness over time
                return self.current_iteration

            @staticmethod
            def get_adjusted_fitness():
                # Return a constant adjusted fitness value (not relevant for stopping condition in this test)
                return 0.0

            def find_neighbors(self):
                # Simulate finding neighbors (not used in this specific test)
                pass

            def best_neighbor(self):
                # Simulate finding the best neighbor by updating part of the state based on the current iteration
                self.state[: self.current_iteration + 1] = 1
                return self.state

            def eval_fitness(self, state):
                # Evaluate the fitness as the sum of the state array
                return np.sum(state)

            def get_length(self):
                # Return the length of the state array
                return len(self.state)

            def get_maximize(self):
                # Return whether the problem is a maximization problem
                return self.maximize

            def can_stop(self):
                # Return `True` to trigger stopping the algorithm after 2 iterations
                return self.current_iteration >= early_stop_iter

        problem = MockProblem()

        # Run the hill climbing algorithm with the mock problem, allowing up to 10 iterations
        # Expect the algorithm to stop early due to `can_stop()` after 2 iterations
        best_state, best_fitness, fitness_curve = hill_climb(problem=problem, max_iters=10, curve=True)

        assert best_state is not None
        assert problem.current_iteration == early_stop_iter
        assert isinstance(best_fitness, float)
        assert isinstance(fitness_curve, np.ndarray)
