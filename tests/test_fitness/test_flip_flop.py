"""Unit tests for fitness/flip_flop.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import pytest
import numpy as np

from mlrose_ky import FlipFlop


class TestFlipFlop:
    """Unit tests for FlipFlop."""

    def test_flip_flop_evaluate_with_invalid_state(self):
        """Test that FlipFlop evaluate raises TypeError when state is invalid."""
        state = [0, 1, 0, 1, 1, 1, 1]
        with pytest.raises(TypeError, match=f"Expected state to be np.ndarray, got {type(state).__name__} instead."):
            # noinspection PyTypeChecker
            _ = FlipFlop().evaluate(state)

    def test_flip_flop_evaluate__many_with_invalid_states(self):
        """Test that FlipFlop evaluate_many raises TypeError when states is invalid."""
        states = [[0, 1, 0, 1, 1, 1, 1]]
        with pytest.raises(TypeError, match=f"Expected states matrix to be np.ndarray, got {type(states).__name__} instead."):
            # noinspection PyTypeChecker
            _ = FlipFlop().evaluate_many(states)

    def test_flipflop(self):
        """Test FlipFlop fitness function."""
        state = np.array([0, 1, 0, 1, 1, 1, 1])
        assert FlipFlop().evaluate(state) == 3
