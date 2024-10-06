"""Unit tests for opt_probs/tsp_opt.py"""

import re

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np
import pytest

from mlrose_ky import OneMax
from mlrose_ky.opt_probs import TSPOpt


class TestTSPOpt:
    """Tests for TSPOpt class."""

    def test_init_no_params(self):
        """Test __init__ when no fitness_fn, coords, or distances are provided."""
        with pytest.raises(ValueError, match=re.escape("At least one of fitness_fn, coords, or distances must be specified.")):
            _ = TSPOpt()

    def test_init_invalid_fitness_fn(self):
        """Test __init__ when fitness_fn does not have prob_type 'tsp'."""
        fitness_fn = OneMax()
        with pytest.raises(ValueError, match=re.escape("fitness_fn must have problem type 'tsp'.")):
            _ = TSPOpt(length=5, fitness_fn=fitness_fn)

    def test_sample_pop_negative(self):
        """Test sample_pop method with negative sample_size."""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7)]
        problem = TSPOpt(5, distances=dists)
        with pytest.raises(ValueError, match=re.escape("sample_size must be a positive integer, got -1.")):
            problem.sample_pop(-1)

    def test_sample_pop_float_non_integer(self):
        """Test sample_pop with sample_size as a non-integer float."""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7)]
        problem = TSPOpt(5, distances=dists)
        with pytest.raises(ValueError, match=re.escape("sample_size must be a positive integer, got 2.5.")):
            # noinspection PyTypeChecker
            problem.sample_pop(2.5)

    def test_sample_pop_float_integer(self):
        """Test sample_pop with sample_size as a float."""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7)]
        problem = TSPOpt(5, distances=dists)
        problem.keep_sample = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        problem.eval_node_probs()
        with pytest.raises(ValueError, match=re.escape("sample_size must be a positive integer, got 2.0.")):
            # noinspection PyTypeChecker
            _ = problem.sample_pop(2.0)

    def test_adjust_probs_all_zero(self):
        """Test adjust_probs method when all elements in input vector sum to zero."""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        probs = np.zeros(5)
        assert np.array_equal(problem.adjust_probs(probs), np.zeros(5))

    def test_adjust_probs_non_zero(self):
        """Test adjust_probs method when all elements in input vector sum to some value other than zero."""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        probs = np.array([0.1, 0.2, 0, 0, 0.5])
        x = np.array([0.125, 0.25, 0, 0, 0.625])
        assert np.array_equal(problem.adjust_probs(probs), x)

    def test_find_neighbors(self):
        """Test find_neighbors method"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        problem.find_neighbors()
        neigh = np.array(
            [
                [1, 0, 2, 3, 4],
                [2, 1, 0, 3, 4],
                [3, 1, 2, 0, 4],
                [4, 1, 2, 3, 0],
                [0, 2, 1, 3, 4],
                [0, 3, 2, 1, 4],
                [0, 4, 2, 3, 1],
                [0, 1, 3, 2, 4],
                [0, 1, 4, 3, 2],
                [0, 1, 2, 4, 3],
            ]
        )
        assert np.array_equal(np.array(problem.neighbors), neigh)

    def test_random(self):
        """Test random method"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        rand = problem.random()
        assert len(rand) == 5 and len(set(rand)) == 5

    def test_random_mimic(self):
        """Test random_mimic method"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        pop = np.array([[1, 0, 3, 2, 4], [0, 2, 1, 3, 4], [0, 2, 4, 3, 1], [4, 1, 3, 2, 0], [3, 4, 0, 2, 1], [2, 4, 0, 3, 1]])
        problem = TSPOpt(5, distances=dists)
        problem.keep_sample = pop
        problem.eval_node_probs()
        problem.find_sample_order()
        rand = problem.random_mimic()
        assert len(rand) == 5 and len(set(rand)) == 5

    def test_random_neighbor(self):
        """Test random_neighbor method"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        neigh = problem.random_neighbor()
        abs_diff = np.abs(x - neigh)
        abs_diff[abs_diff > 0] = 1
        sum_diff = np.sum(abs_diff)
        assert len(neigh) == 5 and sum_diff == 2 and len(set(neigh)) == 5

    def test_reproduce_mut0(self):
        """Test reproduce method when mutation_prob is 0"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        father = np.array([0, 1, 2, 3, 4])
        mother = np.array([0, 4, 3, 2, 1])
        child = problem.reproduce(father, mother, mutation_prob=0)
        assert len(child) == 5 and len(set(child)) == 5

    def test_reproduce_mut1(self):
        """Test reproduce method when mutation_prob is 1"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        problem = TSPOpt(5, distances=dists)
        father = np.array([0, 1, 2, 3, 4])
        mother = np.array([4, 3, 2, 1, 0])
        child = problem.reproduce(father, mother, mutation_prob=1)
        assert len(child) == 5 and len(set(child)) == 5

    def test_sample_pop(self):
        """Test sample_pop method"""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        pop = np.array([[1, 0, 3, 2, 4], [0, 2, 1, 3, 4], [0, 2, 4, 3, 1], [4, 1, 3, 2, 0], [3, 4, 0, 2, 1], [2, 4, 0, 3, 1]])
        problem = TSPOpt(5, distances=dists)
        problem.keep_sample = pop
        problem.eval_node_probs()
        sample = problem.sample_pop(100)
        row_sums = np.sum(sample, axis=1)
        assert np.shape(sample)[0] == 100 and np.shape(sample)[1] == 5 and max(row_sums) == 10 and min(row_sums) == 10

    def test_init_with_distances_only(self):
        """Test __init__ when distances are provided but coords and length are None."""
        distances = [(0, 1, 1), (1, 2, 2), (2, 0, 3)]
        problem = TSPOpt(distances=distances)
        assert problem.length == 3
