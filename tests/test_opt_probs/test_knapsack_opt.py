"""Unit tests for opt_probs/knapsack_opt.py"""

import re

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky import FlipFlop, Knapsack
from mlrose_ky.opt_probs import KnapsackOpt


class TestKnapsackOpt:
    """Tests for KnapsackOpt class."""

    def test_knapsack_opt_invalid_weights_values(self):
        """Test that KnapsackOpt raises ValueError when either weights or values is None."""
        weights = [10, 5, 2]
        values = [1, 2, 3]
        with pytest.raises(ValueError, match="Either fitness_fn or both weights and values must be specified."):
            _ = KnapsackOpt(weights=weights)
        with pytest.raises(ValueError, match="Either fitness_fn or both weights and values must be specified."):
            _ = KnapsackOpt(values=values)
        with pytest.raises(ValueError, match="Either fitness_fn or both weights and values must be specified."):
            _ = KnapsackOpt(weights=[], values=[])

    def test_knapsack_opt_invalid_max_weight_pct(self):
        """Test that KnapsackOpt raises ValueError when max_weight_pct has an invalid value."""
        max_weight_pct = -0.1
        with pytest.raises(
            ValueError, match=re.escape(f"max_weight_pct must be between 0 (inclusive) and 1 (exclusive), got {max_weight_pct}.")
        ):
            _ = KnapsackOpt(weights=[1], values=[1], max_weight_pct=max_weight_pct)

    def test_knapsack_opt_with_weights_and_values(self):
        """Test that KnapsackOpt can be initialized with weights and values."""
        weights = [10, 5, 2]
        values = [1, 2, 3]
        problem = KnapsackOpt(weights=weights, values=values)
        assert problem.length == 3
        assert problem.max_val == 2

    def test_knapsack_opt_initialization_with_fitness_fn(self):
        """Test that KnapsackOpt can be initialized with a Knapsack fitness function."""
        fitness_fn = Knapsack(weights=[10, 5, 2], values=[1, 2, 3])
        problem = KnapsackOpt(fitness_fn=fitness_fn)
        assert problem.length == 3
        assert problem.max_val == 2

    def test_knapsack_opt_set_state(self):
        """Test set_state method"""
        weights = [10, 5, 2]
        values = [1, 2, 3]
        problem = KnapsackOpt(weights=weights, values=values, max_weight_pct=0.5)
        state = np.array([1, 0, 1])
        problem.set_state(state)
        assert np.array_equal(problem.get_state().tolist(), state)

    def test_knapsack_opt_eval_fitness(self):
        """Test eval_fitness method"""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        problem = KnapsackOpt(weights=weights, values=values, max_weight_pct=0.6)
        state = np.array([1, 0, 2, 1, 0])
        fitness = problem.eval_fitness(state)
        assert fitness == 11.0  # Assuming the fitness function calculates correctly

    def test_knapsack_opt_set_population(self):
        """Test set_population method"""
        weights = [10, 5, 2]
        values = [1, 2, 3]
        problem = KnapsackOpt(weights=weights, values=values, max_weight_pct=0.5)
        pop = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        problem.set_population(pop)
        assert np.array_equal(problem.get_population().tolist(), pop)
