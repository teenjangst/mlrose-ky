"""Unit tests for fitness/_discrete_peaks_base.py"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest

# noinspection PyProtectedMember
from mlrose_ky.fitness._discrete_peaks_base import _DiscretePeaksBase


class TestDiscretePeaksBase:
    """Unit tests for _DiscretePeaksBase."""

    def test_head_with_invalid_number(self):
        """Test head function with invalid head number."""
        state = np.array([1, 1, 1, 1])
        with pytest.raises(TypeError, match="Expected number to be an int, got float instead."):
            # noinspection PyTypeChecker
            _DiscretePeaksBase.head(1.5, state)

    def test_head_with_invalid_state(self):
        """Test head function with invalid state."""
        state = [1, 1, 1, 1]
        with pytest.raises(TypeError, match="Expected vector to be np.ndarray, got list instead."):
            # noinspection PyTypeChecker
            _DiscretePeaksBase.head(1, state)

    def test_tail_with_invalid_number(self):
        """Test tail function with invalid tail number."""
        state = np.array([1, 1, 1, 1])
        with pytest.raises(TypeError, match="Expected number to be an int, got float instead."):
            # noinspection PyTypeChecker
            _DiscretePeaksBase.tail(1.5, state)

    def test_tail_with_invalid_state(self):
        """Test head function with invalid state."""
        state = [1, 1, 1, 1]
        with pytest.raises(TypeError, match="Expected vector to be np.ndarray, got list instead."):
            # noinspection PyTypeChecker
            _DiscretePeaksBase.tail(1, state)

    def test_head(self):
        """Test head function"""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert _DiscretePeaksBase.head(1, state) == 4

    def test_tail(self):
        """Test tail function"""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert _DiscretePeaksBase.tail(1, state) == 2
