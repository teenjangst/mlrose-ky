"""Unit tests for opt_probs/max_k_color_opt.py"""

import re

# Author: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import numpy as np
import pytest

import mlrose_ky
from mlrose_ky.opt_probs import MaxKColorOpt


class TestMaxKColorOpt:
    """Tests for MaxKColorOpt class."""

    def test_max_k_color_invalid_initialization(self):
        """Test that MaxKColor raises errors when invalid parameters are used."""
        with pytest.raises(ValueError, match=re.escape("Either fitness_fn or edges must be specified.")):
            _ = MaxKColorOpt()
        with pytest.raises(AttributeError, match=re.escape("Expected fitness_fn to have a weights attribute.")):
            _ = MaxKColorOpt(fitness_fn={})
        with pytest.raises(ValueError, match=re.escape("Expected length to be a non-negative int greater than 0, got 0.")):
            _ = MaxKColorOpt(edges=[(0, 1)], length=0)

    # Commented out test_initialization_inferred_length because unsure how problem should behave when !edges && fitness_fn.weights.
    # def test_initialization_inferred_length(self):
    #     """Test that MaxKColor is initialized correctly when length is inferred from edges or weights."""
    #     fitness_fn = FlipFlop()
    #     fitness_fn.weights = np.ones(5).tolist()
    #     queens_opt = MaxKColorOpt(fitness_fn=fitness_fn)
    #
    #     assert queens_opt.length == 5
    #     assert queens_opt.max_val == 5
    #     assert queens_opt.stop_fitness == 0

    def test_add_missing_nodes(self):
        """Test that missing nodes are added to the graph when edges don't cover all nodes."""
        edges = [(0, 1)]  # Only two nodes connected by an edge
        length = 4  # Expecting 4 nodes in total, so nodes 2 and 3 should be added

        max_k_color_opt = MaxKColorOpt(edges=edges, length=length)

        # Check if all nodes up to `length` have been added
        assert max_k_color_opt.source_graph.number_of_nodes() == length
        assert all(node in max_k_color_opt.source_graph.nodes for node in range(length))

        # Check that the edges are still correct
        assert max_k_color_opt.source_graph.number_of_edges() == len(edges)
        assert all(edge in max_k_color_opt.source_graph.edges for edge in edges)

    def test_initialization(self):
        """Test that the MaxKColorOpt class is initialized correctly."""
        edges = [(0, 1), (1, 2), (2, 0)]
        length = 3
        max_k_color_opt = MaxKColorOpt(edges=edges, length=length)

        assert max_k_color_opt.length == length
        assert isinstance(max_k_color_opt.fitness_fn, mlrose_ky.MaxKColor)
        assert max_k_color_opt.max_val == 3  # In a triangle graph, you need 3 colors
        assert max_k_color_opt.stop_fitness == 0  # Minimize conflicts

    def test_state_initialization(self):
        """Test that the initial state is correctly set."""
        edges = [(0, 1), (1, 2), (2, 0)]
        max_k_color_opt = MaxKColorOpt(edges=edges)
        state = max_k_color_opt.get_state()

        assert len(state) == len(edges)
        assert np.all((0 <= state) & (state < max_k_color_opt.max_val))

    def test_set_state(self):
        """Test that setting a new state updates fitness correctly."""
        edges = [(0, 1), (1, 2), (2, 0)]
        max_k_color_opt = MaxKColorOpt(edges=edges)
        new_state = np.array([0, 1, 2])
        max_k_color_opt.set_state(new_state)

        assert np.array_equal(max_k_color_opt.get_state(), new_state)
        assert max_k_color_opt.get_fitness() == 0  # No adjacent nodes with the same color

    def test_can_stop(self):
        """Test the can_stop method works correctly."""
        edges = [(0, 1), (1, 2), (2, 0)]
        max_k_color_opt = MaxKColorOpt(edges=edges)

        # State with all nodes the same color should not stop (maximization mode)
        max_k_color_opt.set_state(np.array([0, 0, 0]))
        assert not max_k_color_opt.can_stop()

        # State with no adjacent nodes of the same color should stop
        max_k_color_opt.set_state(np.array([0, 1, 2]))
        assert max_k_color_opt.can_stop()

    def test_random_state_generation(self):
        """Test that random state generation produces valid states."""
        edges = [(0, 1), (1, 2), (2, 0)]
        max_k_color_opt = MaxKColorOpt(edges=edges)
        random_state = max_k_color_opt.random()

        assert len(random_state) == 3
        assert np.all((0 <= random_state) & (random_state < 3))

    def test_fitness_evaluation_suboptimal_state(self):
        """Test fitness evaluation for a sub-optimal state."""
        edges = [(0, 1), (1, 2), (2, 0)]
        max_k_color_opt = MaxKColorOpt(edges=edges)
        state = np.array([0, 0, 0])  # All nodes have the same color
        max_k_color_opt.set_state(state)

        assert max_k_color_opt.get_fitness() == -3  # 3 pairs of adjacent nodes with the same color

    def test_fitness_evaluation_optimal_state(self):
        """Test fitness evaluation for an optimal state."""
        edges = [(0, 1), (1, 2), (2, 0)]
        max_k_color_opt = MaxKColorOpt(edges=edges)
        state = np.array([0, 1, 2])  # No two adjacent nodes have the same color
        max_k_color_opt.set_state(state)

        assert max_k_color_opt.get_fitness() == 0  # No conflicts in minimization mode

    def test_edge_case_single_node(self):
        """Test edge case with a single-node graph."""
        edges = []  # No edges in a single-node graph
        max_k_color_opt = MaxKColorOpt(edges=edges, length=1)

        assert max_k_color_opt.get_fitness() == 0
        assert max_k_color_opt.can_stop()

    def test_edge_case_no_edges(self):
        """Test edge case with no edges in the graph."""
        edges = []
        max_k_color_opt = MaxKColorOpt(edges=edges)

        assert max_k_color_opt.get_fitness() == 0
        assert max_k_color_opt.can_stop()

    def test_maximization_mode(self):
        """Test MaxKColorOpt in maximization mode."""
        edges = [(0, 1), (1, 2), (2, 0)]
        max_k_color_opt = MaxKColorOpt(edges=edges, maximize=True)

        state = np.array([0, 1, 2])  # No conflicts, optimal for maximization mode
        max_k_color_opt.set_state(state)

        assert max_k_color_opt.get_fitness() == 3
