"""Unit tests for fitness/one_max.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import pytest
import numpy as np

from mlrose_ky import OneMax


class TestOneMax:
    """Unit tests for OneMax."""

    def test_one_max_evaluate_invalid_state(self):
        """Test that OneMax evaluate raises TypeError when state is invalid."""
        state = [0, 1, 0, 1, 1, 1, 1]
        with pytest.raises(TypeError, match=f"Expected state to be np.ndarray, got list instead."):
            # noinspection PyTypeChecker
            _ = OneMax().evaluate(state)

    def test_onemax(self):
        """Test OneMax fitness function"""
        state = np.array([0, 1, 0, 1, 1, 1, 1])
        assert OneMax().evaluate(state) == 5
