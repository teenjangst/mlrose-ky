"""Unit tests for algorithms/mimic.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest
import re

from mlrose_ky import DiscreteOpt, ContinuousOpt, OneMax, CustomFitness
from mlrose_ky.algorithms import mimic
from tests.globals import SEED


class TestMimic:
    """Unit tests for mimic."""

    def test_mimic_invalid_noise_value(self):
        noise = 1
        with pytest.raises(ValueError, match=re.escape(f"noise must be between 0 and 0.1 (inclusive). Got {noise}")):
            problem = DiscreteOpt(5, OneMax())
            mimic(problem, random_state=SEED, noise=noise)

    def test_mimic_invalid_pop_size(self):
        """Test that mimic raises ValueError when pop_size is invalid."""
        problem = DiscreteOpt(length=5, fitness_fn=OneMax())
        pop_size = -1
        with pytest.raises(ValueError, match=f"pop_size must be a positive integer greater than 0. Got {pop_size}"):
            mimic(problem, random_state=SEED, pop_size=pop_size)

    def test_mimic_invalid_keep_pct(self):
        """Test that mimic raises ValueError when keep_pct is invalid."""
        problem = DiscreteOpt(length=5, fitness_fn=OneMax())
        keep_pct = -0.1
        with pytest.raises(ValueError, match=re.escape(f"keep_pct must be between 0 and 1 (exclusive). Got {keep_pct}")):
            mimic(problem, random_state=SEED, keep_pct=keep_pct)

    def test_mimic_invalid_max_attempts(self):
        """Test that mimic raises ValueError when max_attempts is invalid."""
        problem = DiscreteOpt(length=5, fitness_fn=OneMax())
        max_attempts = -1
        with pytest.raises(ValueError, match=f"max_attempts must be a positive integer greater than 0. Got {max_attempts}"):
            mimic(problem, random_state=SEED, max_attempts=max_attempts)

    def test_mimic_invalid_max_iters(self):
        """Test that mimic raises ValueError when max_iters is invalid."""
        problem = DiscreteOpt(length=5, fitness_fn=OneMax())
        max_iters = -1
        with pytest.raises(ValueError, match=f"max_iters must be a positive integer greater than 0 or np.inf. Got {max_iters}"):
            mimic(problem, random_state=SEED, max_iters=max_iters)

    def test_mimic_invalid_problem_type(self):
        """Test that mimic raises ValueError when problem type is continuous."""
        fitness = CustomFitness(lambda x: sum(x))
        problem = ContinuousOpt(length=5, fitness_fn=fitness)
        with pytest.raises(ValueError, match="MIMIC algorithm cannot be used for continuous optimization problems."):
            mimic(problem, random_state=SEED)

    def test_mimic_invalid_callback_user_info_type(self):
        """Test that mimic raises TypeError when callback_user_info is not a dict."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(TypeError, match="callback_user_info must be a dict. Got str"):
            # noinspection PyTypeChecker
            mimic(problem, callback_user_info="Invalid callback data")

    def test_mimic_discrete_max(self):
        """Test mimic function for a discrete maximization problem"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = mimic(problem, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_mimic_discrete_min(self):
        """Test mimic function for a discrete minimization problem"""
        problem = DiscreteOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = mimic(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_mimic_callback_early_termination(self):
        """Test mimic with early termination via state_fitness_callback when callback_user_info is None"""
        problem = DiscreteOpt(5, OneMax())

        def callback_function(iteration, attempt, done, state, fitness, fitness_evaluations, curve, user_data):
            return False  # Terminate immediately

        best_state, best_fitness, _ = mimic(problem, state_fitness_callback=callback_function, random_state=SEED)
        # Verify that the best_state is valid
        assert isinstance(best_state, np.ndarray)
        assert isinstance(best_fitness, float)

    def test_mimic_problem_can_stop(self):
        """Test mimic where problem.can_stop() returns True"""

        class TestProblem(DiscreteOpt):
            def can_stop(self):
                return True

        problem = TestProblem(5, OneMax())
        best_state, best_fitness, _ = mimic(problem, random_state=SEED)
        # Since can_stop() returns True, the algorithm should terminate immediately
        assert isinstance(best_state, np.ndarray)
        assert isinstance(best_fitness, float)
