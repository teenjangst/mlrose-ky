"""Unit tests for fitness/custom_fitness.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import re

import pytest
import numpy as np

from mlrose_ky import CustomFitness


class TestCustomFitness:
    """Unit tests for CustomFitness."""

    def test_custom_fitness_with_invalid_fitness_fn(self):
        """Test CustomFitness initialized with invalid fitness function."""
        fitness_fn = {}  # fitness_fn should be a callable function

        with pytest.raises(
            TypeError, match=re.escape(f"Expected fitness_fn to be a callable function, got {type(fitness_fn).__name__} instead.")
        ):
            # noinspection PyTypeChecker
            _ = CustomFitness(fitness_fn)

    def test_custom_fitness_with_invalid_problem_type(self):
        """Test CustomFitness initialized with invalid problem type."""

        # noinspection PyMissingOrEmptyDocstring
        def custom_fitness(_state, c):
            return c * np.sum(_state)

        problem_type = "invalid_problem_type"
        state = [1, 2, 3, 4]

        with pytest.raises(
            ValueError,
            match=re.escape(f"Invalid problem_type: {problem_type}. Must be one of ['discrete', 'continuous', 'tsp', 'either']."),
        ):
            # noinspection PyTypeChecker
            assert CustomFitness(custom_fitness, problem_type=problem_type).evaluate(state) == 150

    def test_custom_fitness_evaluate_with_invalid_state(self):
        """Test CustomFitness evaluate called with invalid state."""

        # noinspection PyMissingOrEmptyDocstring
        def custom_fitness(_state, c):
            return c * np.sum(_state)

        state = [1, 2, 3, 4]
        with pytest.raises(TypeError, match=re.escape(f"Expected state to be np.ndarray, got list instead.")):
            # noinspection PyTypeChecker
            assert CustomFitness(custom_fitness).evaluate(state) == 150

    def test_custom_fitness(self):
        """Test CustomFitness fitness function"""

        # noinspection PyMissingOrEmptyDocstring
        def custom_fitness(_state, c):
            return c * np.sum(_state)

        state = np.array([1, 2, 3, 4, 5])
        kwargs = {"c": 10}
        assert CustomFitness(custom_fitness, **kwargs).evaluate(state) == 150
