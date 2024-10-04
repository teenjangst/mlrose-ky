"""Unit tests for fitness/six_peaks.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import re

import numpy as np
import pytest

from mlrose_ky import SixPeaks


class TestSixPeaks:
    """Unit tests for SixPeaks."""

    def test_six_peaks_invalid_t_pct(self):
        """Test that SixPeaks raises ValueError when t_pct is invalid."""
        with pytest.raises(ValueError, match=re.escape(f"threshold_pct must be between 0 and 1 (inclusive), got -0.1.")):
            SixPeaks(t_pct=-0.1)
        with pytest.raises(ValueError, match=re.escape(f"threshold_pct must be between 0 and 1 (inclusive), got 1.1.")):
            SixPeaks(t_pct=1.1)

    def test_six_peaks_evaluate_invalid_state(self):
        """Test that SixPeaks evaluate raises TypeError when state is invalid."""
        state = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        with pytest.raises(TypeError, match=re.escape(f"Expected state_vector to be np.ndarray, got list instead.")):
            # noinspection PyTypeChecker
            SixPeaks().evaluate(state)

    def test_sixpeaks_r0(self):
        """Test SixPeaks fitness function for the case where R=0 and max>0"""
        state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        assert SixPeaks(t_pct=0.30).evaluate(state) == 4

    def test_sixpeaks_r_gt0(self):
        """Test SixPeaks fitness function for the case where R>0 and max>0"""
        state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        assert SixPeaks(t_pct=0.15).evaluate(state) == 16

    def test_sixpeaks_r0_max0(self):
        """Test SixPeaks fitness function for the case where R=0 and max=0"""
        state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
        assert SixPeaks(t_pct=0.30).evaluate(state) == 0

    def test_sixpeaks_r_gt0_max0(self):
        """Test SixPeaks fitness function for the case where R>0 and max=0"""
        state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
        assert SixPeaks(t_pct=0.15).evaluate(state) == 12
