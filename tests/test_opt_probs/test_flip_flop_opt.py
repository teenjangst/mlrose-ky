"""Unit tests for opt_probs/test_flip_flop_opt.py"""

import re

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky import FlipFlop
from mlrose_ky.opt_probs import FlipFlopOpt


class TestFlipFlopOpt:
    """Tests for FlipFlopOpt class."""

    def test_invalid_inputs(self):
        """Test methods with invalid inputs"""
        problem_size, invalid_size = 5, 3
        problem = FlipFlopOpt(problem_size)
        invalid_state = np.ones((invalid_size,))

        with pytest.raises(ValueError, match=re.escape(f"new_state length {invalid_size} must match problem length {problem_size}")):
            problem.set_state(invalid_state)

        with pytest.raises(ValueError, match=re.escape("pop_size must be a positive, non-zero integer. Got -1.")):
            problem.random_pop(-1)

    def test_raise_value_error_when_fitness_fn_and_length_not_specified(self):
        with pytest.raises(ValueError, match=re.escape("fitness_fn or length must be specified.")):
            FlipFlopOpt()

    def test_random_pop_edge_cases(self):
        """Test random_pop method with edge cases"""
        problem = FlipFlopOpt(5)

        with pytest.raises(ValueError, match=re.escape("pop_size must be a positive, non-zero integer. Got 0.")):
            problem.random_pop(0)

        # Test with a large population size
        problem.random_pop(10000)
        pop = problem.get_population()
        assert pop.shape == (10000, 5) and np.all((pop == 0) | (pop == 1))

    def test_length_inference_from_fitness_fn(self):
        fitness_fn = FlipFlop()
        fitness_fn.weights = [1, 0, 1, 0, 1]
        opt = FlipFlopOpt(fitness_fn=fitness_fn)
        assert opt.length == len(fitness_fn.weights)

    def test_set_state(self):
        """Test set_state method"""
        problem = FlipFlopOpt(5)
        x = np.array([0, 1, 0, 1, 0])
        problem.set_state(x)
        assert np.array_equal(problem.get_state(), x)

    def test_set_population(self):
        """Test set_population method"""
        problem = FlipFlopOpt(5)
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.set_population(pop)
        assert np.array_equal(problem.get_population(), pop)

    def test_best_child(self):
        """Test best_child method"""
        problem = FlipFlopOpt(5)
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.set_population(pop)
        x = problem.best_child()
        assert np.array_equal(x, np.array([1, 0, 1, 0, 1]))

    def test_best_neighbor(self):
        """Test best_neighbor method"""
        problem = FlipFlopOpt(5)
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.neighbors = pop
        x = problem.best_neighbor()
        assert np.array_equal(x, np.array([1, 0, 1, 0, 1]))

    def test_evaluate_population_fitness(self):
        """Test evaluate_population_fitness method"""
        problem = FlipFlopOpt(5)
        pop = np.array([[0, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        problem.set_population(pop)
        problem.evaluate_population_fitness()
        expected_fitness = np.array([1, 4, 1, 2, 0, 0])
        assert np.array_equal(problem.get_pop_fitness(), expected_fitness)

    def test_random_pop(self):
        """Test random_pop method"""
        problem = FlipFlopOpt(5)
        problem.random_pop(10)
        pop = problem.get_population()
        assert pop.shape == (10, 5) and np.all((pop == 0) | (pop == 1))

    def test_set_state_boundary_conditions(self):
        """Test set_state method with boundary conditions"""
        problem = FlipFlopOpt(5)
        # Test with minimum state vector
        min_state = np.array([0, 0, 0, 0, 0])
        problem.set_state(min_state)
        assert np.array_equal(problem.get_state(), min_state)

        # Test with maximum state vector
        max_state = np.array([1, 1, 1, 1, 1])
        problem.set_state(max_state)
        assert np.array_equal(problem.get_state(), max_state)

    def test_can_stop_with_sub_optimal_state(self):
        """Test can_stop method given a sub-optimal state"""
        problem = FlipFlopOpt(5)
        problem.set_state(np.array([1, 1, 1, 1, 1]))
        assert not problem.can_stop()

    def test_can_stop_with_optimal_state(self):
        problem = FlipFlopOpt(5)
        problem.set_state(np.array([1, 0, 1, 0, 1]))
        assert problem.can_stop()
