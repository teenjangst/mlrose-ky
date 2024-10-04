"""Unit tests for fitness/four_peaks.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import pytest
import numpy as np

from mlrose_ky import FourPeaks


class TestFourPeaks:
    """Unit tests for FourPeaks."""

    def test_four_peaks_invalid_t_pct(self):
        """Test that FourPeaks raises ValueError when t_pct is invalid."""
        with pytest.raises(ValueError, match=f"threshold_pct must be between 0 and 1, got -0.1."):
            _ = FourPeaks(t_pct=-0.1)
        with pytest.raises(ValueError, match=f"threshold_pct must be between 0 and 1, got 1.1."):
            _ = FourPeaks(t_pct=1.1)

    def test_four_peaks_evaluate_invalid_state(self):
        """Test that FourPeaks evaluate raises TypeError when state is invalid."""
        state = [1, 1, 1, 0]
        with pytest.raises(TypeError, match=f"Expected state to be np.ndarray, got {type(state).__name__} instead."):
            # noinspection PyTypeChecker
            _ = FourPeaks().evaluate(state)

    def test_fourpeaks_r0(self):
        """Test FourPeaks fitness function for the case where R=0 and max>0"""
        state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        assert FourPeaks(t_pct=0.30).evaluate(state) == 4

    def test_fourpeaks_r_gt0(self):
        """Test FourPeaks fitness function for the case where R>0 and max>0"""
        state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        assert FourPeaks(t_pct=0.15).evaluate(state) == 16

    def test_fourpeaks_r0_max0(self):
        """Test FourPeaks fitness function for the case where R=0 and max=0"""
        state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
        assert FourPeaks(t_pct=0.30).evaluate(state) == 0
