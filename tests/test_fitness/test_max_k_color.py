"""Unit tests for fitness/max_k_color.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky import MaxKColor


class TestMaxKColor:
    """Unit tests for MaxKColor."""

    def test_max_k_color_invalid_edges(self):
        """Test that MaxKColor raises TypeError when edges is invalid."""
        edges = [0, 1, 2, 3, 4, 5]
        with pytest.raises(TypeError, match=f"Expected edges to be a list of tuples of ints."):
            # noinspection PyTypeChecker
            _ = MaxKColor(edges)
        with pytest.raises(TypeError, match=f"Expected edges to be a list of tuples of ints."):
            # noinspection PyTypeChecker
            _ = MaxKColor(np.asarray(edges))

    def test_max_k_color_evaluate_invalid_state(self):
        """Test that MaxKColor evaluate raises TypeError when state is invalid."""
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4), (0, 5)]
        state = [0, 1, 0, 1, 1, 1]
        with pytest.raises(TypeError, match=f"Expected state to be np.ndarray, got list instead."):
            # noinspection PyTypeChecker
            _ = MaxKColor(edges).evaluate(state)

    def test_max_k_color(self):
        """Test MaxKColor fitness function"""
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4), (0, 5)]
        state = np.array([0, 1, 0, 1, 1, 1])
        assert MaxKColor(edges).evaluate(state) == 3
