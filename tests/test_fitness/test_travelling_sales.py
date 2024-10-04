"""Unit tests for fitness/travelling_sales.py"""

# Authors: Kyle Nakamura
# License: BSD 3-clause

import re

import numpy as np
import pytest

from mlrose_ky import TravellingSales


class TestTravellingSales:
    """Unit tests for TravellingSales."""

    def test_travelling_sales_no_coords_no_distances(self):
        """Test that TravellingSales raises ValueError when neither coords nor distances are provided."""
        with pytest.raises(ValueError, match="At least one of coords and distances must be specified."):
            TravellingSales()

    def test_travelling_sales_negative_distance(self):
        """Test that TravellingSales raises ValueError when a distance is negative or zero."""
        distances = [(0, 1, -5), (1, 2, 3)]
        with pytest.raises(ValueError, match="The distance between each pair of nodes must be greater than 0."):
            TravellingSales(distances=distances)

    def test_travelling_sales_zero_distance(self):
        """Test that TravellingSales raises ValueError when a distance is zero."""
        distances = [(0, 1, 0), (1, 2, 3)]
        with pytest.raises(ValueError, match="The distance between each pair of nodes must be greater than 0."):
            TravellingSales(distances=distances)

    def test_travelling_sales_negative_node(self):
        """Test that TravellingSales raises ValueError when node index is negative."""
        distances = [(-1, 1, 5), (1, 2, 3)]
        with pytest.raises(ValueError, match="The minimum node value must be 0."):
            TravellingSales(distances=distances)

    def test_travelling_sales_missing_nodes(self):
        """Test that TravellingSales raises ValueError when nodes are missing in distances."""
        distances = [(0, 1, 5), (1, 2, 3), (2, 4, 2)]  # Node 3 is missing
        with pytest.raises(ValueError, match="All nodes must appear at least once in distances."):
            TravellingSales(distances=distances)

    def test_travelling_sales_invalid_state_length_coords(self):
        """Test that TravellingSales raises ValueError when state length does not match coords length."""
        coords = [(0, 0), (3, 0), (3, 2)]
        state = np.array([0, 1])  # Incorrect length
        fitness = TravellingSales(coords=coords)
        with pytest.raises(ValueError, match="state must have the same length as coords."):
            fitness.evaluate(state)

    def test_travelling_sales_repeated_nodes(self):
        """Test that TravellingSales raises ValueError when state contains repeated nodes."""
        coords = [(0, 0), (3, 0), (3, 2), (2, 4)]
        state = np.array([0, 1, 2, 2])  # Node 2 is repeated
        fitness = TravellingSales(coords=coords)
        with pytest.raises(ValueError, match="Each node must appear exactly once in state."):
            fitness.evaluate(state)

    def test_travelling_sales_negative_node_in_state(self):
        """Test that TravellingSales raises ValueError when state contains negative node indices."""
        coords = [(0, 0), (3, 0), (3, 2)]
        state = np.array([0, -1, 2])  # Negative node
        fitness = TravellingSales(coords=coords)
        with pytest.raises(ValueError, match="All elements of state must be non-negative integers."):
            fitness.evaluate(state)

    def test_travelling_sales_node_index_out_of_range(self):
        """Test that TravellingSales raises ValueError when state contains node indices out of range."""
        coords = [(0, 0), (3, 0), (3, 2)]
        state = np.array([0, 1, 3])  # Node index 3 out of range
        fitness = TravellingSales(coords=coords)
        with pytest.raises(ValueError, match=re.escape("All elements of state must be less than len(state).")):
            fitness.evaluate(state)

    def test_travelling_sales_get_prob_type(self):
        """Test that TravellingSales get_prob_type method returns 'tsp'."""
        coords = [(0, 0), (3, 0), (3, 2)]
        fitness = TravellingSales(coords=coords)
        assert fitness.get_prob_type() == "tsp"

    def test_travelling_sales_coords(self):
        """Test TravellingSales fitness function for case where city nodes coords are specified."""
        coords = [(0, 0), (3, 0), (3, 2), (2, 4), (1, 3)]
        state = np.array([0, 1, 4, 3, 2])
        fitness_value = TravellingSales(coords=coords).evaluate(state)
        assert round(fitness_value, 4) == 13.8614

    def test_travelling_sales_dists(self):
        """Test TravellingSales fitness function for case where distances between node pairs are specified."""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (1, 4, 9), (2, 3, 8), (2, 4, 2), (3, 4, 4)]
        state = np.array([0, 1, 4, 3, 2])
        fitness_value = TravellingSales(distances=dists).evaluate(state)
        assert fitness_value == 29.0

    def test_travelling_sales_invalid(self):
        """Test TravellingSales fitness function for invalid tour."""
        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6), (1, 4, 9), (2, 3, 8), (2, 4, 2), (3, 4, 4)]
        state = np.array([0, 1, 2, 3, 4])  # There is no direct path between 3 and 4
        fitness_value = TravellingSales(distances=dists).evaluate(state)
        assert fitness_value == np.inf

    def test_travelling_sales_duplicate_distances(self):
        """Test that duplicate distances are handled correctly."""
        dists = [
            (0, 1, 3),
            (1, 0, 3),  # Duplicate entry
            (0, 2, 5),
            (2, 0, 5),  # Duplicate entry
            (0, 3, 1),
            (0, 4, 7),
            (1, 3, 6),
            (1, 4, 9),
            (2, 3, 8),
            (2, 4, 2),
            (3, 4, 4),
        ]
        state = np.array([0, 1, 4, 3, 2])
        fitness_value = TravellingSales(distances=dists).evaluate(state)
        assert fitness_value == 29.0
