"""Unit tests for fitness/queens.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import pytest
import numpy as np

from mlrose_ky import Queens
from tests.globals import SEED


class TestQueens:
    """Unit tests for Queens."""

    def test_queens_invalid_maximize_value(self):
        """Test that Queens raises TypeError when maximize is invalid."""
        with pytest.raises(TypeError, match=f"Expected maximize to be bool, got str instead."):
            # noinspection PyTypeChecker
            _ = Queens(maximize="max")

    def test_queens_evaluate_invalid_state(self):
        """Test that Queens evaluate raises TypeError when state is invalid."""
        state = [1, 4, 1, 3, 5, 5, 2, 7]
        with pytest.raises(TypeError, match=f"Expected state to be np.ndarray, got list instead."):
            # noinspection PyTypeChecker
            _ = Queens().evaluate(state)

    def test_queens_shift_num_is_zero(self):
        """Test that Queens shift returns unchanged state when num is zero."""
        state = np.ndarray([1, 2])
        assert np.array_equal(state.copy(), Queens().shift(state, 0))

    def test_queens_get_max_size_x(self):
        """Test Queens get_max_size with valid problem sizes."""
        assert Queens().get_max_size(0) == 0
        assert Queens().get_max_size(1) == 0
        assert Queens().get_max_size(2) == 1
        assert Queens().get_max_size(3) == 3
        assert Queens().get_max_size(4) == 6

    def test_queens(self):
        """Test Queens fitness function"""
        state = np.array([1, 4, 1, 3, 5, 5, 2, 7])
        fitness = Queens().evaluate(state)
        assert fitness == 6
