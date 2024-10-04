"""Unit tests for fitness/continuous_peaks.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import re

import numpy as np
import pytest

from mlrose_ky import ContinuousPeaks


class TestContinousPeaks:
    """Unit tests for ContinuousPeaks."""

    def test_continuous_peaks_invalid_t_pct(self):
        """Test that ContinuousPeaks raises ValueError when t_pct is invalid."""
        with pytest.raises(ValueError, match=re.escape(f"t_pct must be between 0 and 1 (inclusive), got -0.1 instead.")):
            ContinuousPeaks(t_pct=-0.1)
        with pytest.raises(ValueError, match=re.escape(f"t_pct must be between 0 and 1 (inclusive), got 1.1 instead.")):
            ContinuousPeaks(t_pct=1.1)

    def test_continuous_peaks_evaluate_invalid_state(self):
        """Test that ContinuousPeaks evaluate raises TypeError when state is invalid."""
        state = [1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1]
        with pytest.raises(TypeError, match=re.escape(f"Expected state to be np.ndarray, got list instead.")):
            # noinspection PyTypeChecker
            ContinuousPeaks().evaluate(state)

    def test_max_run_middle(self):
        """Test max_run function for case where run is in the middle of the state."""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert ContinuousPeaks.max_run(1, state) == 5

    def test_max_run_start(self):
        """Test max_run function for case where run is at the start of the state."""
        state = np.array([1, 1, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert ContinuousPeaks.max_run(1, state) == 6

    def test_max_run_end(self):
        """Test max_run function for case where run is at the end of the state."""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert ContinuousPeaks.max_run(1, state) == 9

    def test_max_run_value_not_found(self):
        """Test max_run function for case where the value is not in the state."""
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        assert ContinuousPeaks.max_run(1, state) == 0

    def test_max_run_unfinished_run(self):
        """Test max_run function for case where run continues until the end of the state."""
        state = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        assert ContinuousPeaks.max_run(1, state) == 5

    def test_continuouspeaks_r0(self):
        """Test ContinuousPeaks fitness function for case when R = 0."""
        state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1])
        assert ContinuousPeaks(t_pct=0.30).evaluate(state) == 5

    def test_continuouspeaks_r_gt(self):
        """Test ContinuousPeaks fitness function for case when R > 0."""
        state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1])
        assert ContinuousPeaks(t_pct=0.15).evaluate(state) == 17
