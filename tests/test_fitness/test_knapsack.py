"""Unit tests for fitness/knapsack.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import pytest
import numpy as np

from mlrose_ky import Knapsack


class TestKnapsack:
    """Unit tests for Knapsack."""

    def test_knapsack_invalid_weights_values_length(self):
        """Test that Knapsack raises ValueError when weights and values are of different lengths."""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4]  # Different length
        with pytest.raises(ValueError, match="The weights and values lists must be the same size."):
            Knapsack(weights, values)

    def test_knapsack_negative_weights(self):
        """Test that Knapsack raises ValueError when any weight is less than or equal to zero."""
        weights = [10, -5, 2, 8, 15]  # Negative weight
        values = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="All weights must be greater than 0."):
            Knapsack(weights, values)

    def test_knapsack_negative_values(self):
        """Test that Knapsack raises ValueError when any value is less than or equal to zero."""
        weights = [10, 5, 2, 8, 15]
        values = [1, -2, 3, 4, 5]  # Negative value
        with pytest.raises(ValueError, match="All values must be greater than 0."):
            Knapsack(weights, values)

    def test_knapsack_invalid_max_item_count(self):
        """Test that Knapsack raises ValueError when max_item_count <= 0."""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="max_item_count must be greater than 0."):
            Knapsack(weights, values, max_item_count=0)

    def test_knapsack_invalid_max_weight_pct_zero(self):
        """Test that Knapsack raises ValueError when max_weight_pct <= 0."""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="max_weight_pct must be between 0 and 1."):
            Knapsack(weights, values, max_weight_pct=0)

    def test_knapsack_invalid_max_weight_pct_greater_than_one(self):
        """Test that Knapsack raises ValueError when max_weight_pct > 1."""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="max_weight_pct must be between 0 and 1."):
            Knapsack(weights, values, max_weight_pct=1.5)

    def test_knapsack_evaluate_non_ndarray_state(self):
        """Test that Knapsack raises TypeError when state is not a numpy ndarray."""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        state = [1, 0, 2, 1, 0]  # List instead of ndarray
        fitness = Knapsack(weights, values)
        with pytest.raises(TypeError, match="Expected state_vector to be np.ndarray"):
            # noinspection PyTypeChecker
            fitness.evaluate(state)

    def test_knapsack_evaluate_invalid_state_length(self):
        """Test that Knapsack raises ValueError when state length does not match weights and values length."""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        state = np.array([1, 0, 2])  # Incorrect length
        fitness = Knapsack(weights, values)
        with pytest.raises(ValueError, match="The state_vector must be the same size as the weights and values arrays."):
            fitness.evaluate(state)

    def test_knapsack_evaluate_total_weight_equal_to_max(self):
        """Test Knapsack fitness function when total_weight equals maximum weight (_w)."""
        weights = [2, 4, 6]
        values = [3, 6, 9]
        max_weight_pct = 0.5  # So _w = ceil(12 * 0.5) = 6
        state = np.array([0, 0, 1])  # total_weight = 6
        fitness = Knapsack(weights, values, max_weight_pct)
        result = fitness.evaluate(state)
        assert result == 9.0  # Since total_weight == _w, should return total_value

    def test_knapsack_get_prob_type(self):
        """Test that Knapsack get_prob_type method returns 'discrete'."""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        fitness = Knapsack(weights, values)
        prob_type = fitness.get_prob_type()
        assert prob_type == "discrete"

    def test_knapsack_weight_lt_max(self):
        """Test Knapsack fitness function for case where total weight is less than the maximum."""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        max_weight_pct = 0.6
        state = np.array([1, 0, 2, 1, 0])
        calculated_fitness = Knapsack(weights, values, max_weight_pct).evaluate(state)
        assert calculated_fitness == 11.0

    def test_knapsack_weight_gt_max(self):
        """Test Knapsack fitness function for case where total weight is greater than the maximum."""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        max_weight_pct = 0.4
        state = np.array([1, 0, 2, 1, 0])
        calculated_fitness = Knapsack(weights, values, max_weight_pct).evaluate(state)
        assert calculated_fitness == 0.0
