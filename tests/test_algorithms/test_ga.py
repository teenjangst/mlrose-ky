"""Unit tests for algorithms/ga.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky import DiscreteOpt, OneMax, ContinuousOpt
from mlrose_ky.algorithms import genetic_alg
from tests.globals import SEED


class TestGeneticAlg:
    """Unit tests for genetic_alg."""

    def test_genetic_alg_invalid_pop_size(self):
        """Test that genetic_alg raises ValueError when pop_size is not a positive integer greater than 0."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(ValueError, match="pop_size must be a positive integer greater than 0"):
            genetic_alg(problem, pop_size=-1)

    def test_genetic_alg_invalid_pop_breed_percent(self):
        """Test that genetic_alg raises ValueError when pop_breed_percent is not between 0 and 1 (exclusive)."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(ValueError, match="pop_breed_percent must be between 0 and 1"):
            genetic_alg(problem, pop_breed_percent=1.5)

    def test_genetic_alg_invalid_elite_dreg_ratio(self):
        """Test that genetic_alg raises ValueError when elite_dreg_ratio is not between 0 and 1 (inclusive)."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(ValueError, match="elite_dreg_ratio must be between 0 and 1"):
            genetic_alg(problem, elite_dreg_ratio=-0.1)

    def test_genetic_alg_invalid_minimum_elites(self):
        """Test that genetic_alg raises ValueError when minimum_elites is not a non-negative integer."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(ValueError, match="minimum_elites must be a non-negative integer"):
            genetic_alg(problem, minimum_elites=-1)

    def test_genetic_alg_invalid_minimum_dregs(self):
        """Test that genetic_alg raises ValueError when minimum_dregs is not a non-negative integer."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(ValueError, match="minimum_dregs must be a non-negative integer"):
            genetic_alg(problem, minimum_dregs=-1)

    def test_genetic_alg_invalid_mutation_prob(self):
        """Test that genetic_alg raises ValueError when mutation_prob is not between 0 and 1 (inclusive)."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(ValueError, match="mutation_prob must be between 0 and 1"):
            genetic_alg(problem, mutation_prob=1.5)

    def test_genetic_alg_invalid_max_attempts(self):
        """Test that genetic_alg raises ValueError when max_attempts is not a positive integer greater than 0."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(ValueError, match="max_attempts must be a positive integer greater than 0"):
            genetic_alg(problem, max_attempts=0)

    def test_genetic_alg_invalid_max_iters(self):
        """Test that genetic_alg raises ValueError when max_iters is not a positive integer greater than 0 or np.inf."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(ValueError, match="max_iters must be a positive integer greater than 0 or np.inf"):
            genetic_alg(problem, max_iters=-5)

    def test_genetic_alg_invalid_hamming_factor(self):
        """Test that genetic_alg raises ValueError when hamming_factor is not between 0 and 1 (inclusive)."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(ValueError, match="hamming_factor must be between 0 and 1"):
            genetic_alg(problem, hamming_factor=-0.1)

    def test_genetic_alg_invalid_hamming_decay_factor(self):
        """Test that genetic_alg raises ValueError when hamming_decay_factor is not between 0 and 1 (inclusive)."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(ValueError, match="hamming_decay_factor must be between 0 and 1"):
            genetic_alg(problem, hamming_decay_factor=1.5)

    def test_genetic_alg_invalid_callback_user_info_type(self):
        """Test that genetic_alg raises TypeError when callback_user_info is not a dict."""
        problem = DiscreteOpt(5, OneMax())
        with pytest.raises(TypeError, match=f"callback_user_info must be a dict. Got str"):
            # noinspection PyTypeChecker
            genetic_alg(problem, callback_user_info="User data should be a dict")

    def test_genetic_alg_discrete_max(self):
        """Test genetic_alg function for a discrete maximization problem"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_genetic_alg_continuous_max(self):
        """Test genetic_alg function for a continuous maximization problem"""
        problem = ContinuousOpt(5, OneMax())
        best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
        x = np.ones(5)
        assert np.allclose(best_state, x, atol=0.5) and best_fitness > 4

    def test_genetic_alg_discrete_min(self):
        """Test genetic_alg function for a discrete minimization problem"""
        problem = DiscreteOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.array_equal(best_state, x) and best_fitness == 0

    def test_genetic_alg_continuous_min(self):
        """Test genetic_alg function for a continuous minimization problem"""
        problem = ContinuousOpt(5, OneMax(), maximize=False)
        best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
        x = np.zeros(5)
        assert np.allclose(best_state, x, atol=0.5) and best_fitness < 1

    def test_genetic_alg_callback_early_termination(self):
        """Test genetic_alg with early termination via state_fitness_callback when callback_user_info is None"""
        problem = DiscreteOpt(5, OneMax())

        # noinspection PyMissingOrEmptyDocstring
        def callback_function(iteration, attempt, done, state, fitness, fitness_evaluations, curve, user_data):
            return False  # Terminate immediately

        best_state, best_fitness, _ = genetic_alg(problem, state_fitness_callback=callback_function, random_state=SEED)
        # Since the algorithm terminates immediately, best_state and best_fitness are initial state
        # Verify that the best_state is as expected (since it's random, we can only check the type)
        assert isinstance(best_state, np.ndarray)
        assert isinstance(best_fitness, float)

    def test_genetic_alg_hamming_factor_discrete(self):
        """Test genetic_alg with hamming_factor > 0 for a discrete problem"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = genetic_alg(problem, hamming_factor=0.5, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_genetic_alg_hamming_factor_continuous(self):
        """Test genetic_alg with hamming_factor > 0 for a continuous problem"""
        problem = ContinuousOpt(5, OneMax())
        best_state, best_fitness, _ = genetic_alg(problem, hamming_factor=0.5, random_state=SEED)
        x = np.ones(5)
        assert np.allclose(best_state, x, atol=0.5) and best_fitness > 4

    def test_genetic_alg_hamming_decay_factor(self):
        """Test genetic_alg with hamming_decay_factor specified"""
        problem = DiscreteOpt(5, OneMax())
        best_state, best_fitness, _ = genetic_alg(problem, hamming_factor=0.5, hamming_decay_factor=0.9, random_state=SEED)
        x = np.ones(5)
        assert np.array_equal(best_state, x) and best_fitness == 5

    def test_genetic_alg_breeding_pop_adjustment(self):
        """Test genetic_alg with breeding_pop_size adjustment when dregs_size + elites_size > survivors_size"""
        problem = DiscreteOpt(5, OneMax())
        # Set parameters to force breeding_pop_size adjustment
        best_state, best_fitness, _ = genetic_alg(
            problem,
            pop_size=20,
            pop_breed_percent=0.5,
            minimum_elites=3,
            minimum_dregs=3,
            elite_dreg_ratio=0.0,  # This is key to make dregs_size large
            random_state=SEED,
        )
        # Since breeding_pop_size is adjusted, ensure that algorithm runs and returns valid outputs
        assert isinstance(best_state, np.ndarray)
        assert isinstance(best_fitness, float)
        # We can also check that best_fitness is less than or equal to the maximum possible fitness
        assert best_fitness <= 5

    def test_genetic_alg_negative_breeding_pop_size(self):
        """Test genetic_alg where breeding_pop_size becomes negative and is set to zero"""
        problem = DiscreteOpt(5, OneMax())
        # Set parameters to force breeding_pop_size negative after adjustment
        best_state, best_fitness, _ = genetic_alg(
            problem, pop_size=10, pop_breed_percent=0.1, minimum_elites=8, minimum_dregs=8, elite_dreg_ratio=0.5, random_state=SEED
        )
        # Since breeding_pop_size becomes negative and is set to zero, the line is covered
        # Verify that the algorithm runs and returns valid outputs
        assert isinstance(best_state, np.ndarray)
        assert isinstance(best_fitness, float)

    def test_genetic_alg_problem_can_stop(self):
        """Test genetic_alg where problem.can_stop() returns True"""

        class TestProblem(DiscreteOpt):
            def can_stop(self):
                return True

        problem = TestProblem(5, OneMax())
        best_state, best_fitness, _ = genetic_alg(problem, random_state=SEED)
        # Since can_stop() returns True, the algorithm should terminate immediately
        assert isinstance(best_state, np.ndarray)
        assert isinstance(best_fitness, float)
