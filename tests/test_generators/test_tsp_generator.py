"""Unit tests for generators/"""

import pytest
import numpy as np

from tests.globals import SEED

from mlrose_ky.generators import TSPGenerator


# noinspection PyTypeChecker
class TestTSPGenerator:

    def test_generate_invalid_seed(self):
        """Test generate method raises ValueError when seed is not an integer."""
        with pytest.raises(ValueError, match="Seed must be an integer. Got not_an_int"):
            TSPGenerator.generate(5, seed="not_an_int")

    def test_generate_invalid_number_of_cities(self):
        """Test generate method raises ValueError when number_of_cities is not a positive integer."""
        with pytest.raises(ValueError, match="Number of cities must be a positive integer. Got 0"):
            TSPGenerator.generate(0, seed=SEED)

    def test_generate_invalid_area_width(self):
        """Test generate method raises ValueError when area_width is not a positive integer."""
        with pytest.raises(ValueError, match="Area width must be a positive integer. Got 0"):
            TSPGenerator.generate(5, area_width=0, seed=SEED)

    def test_generate_invalid_area_height(self):
        """Test generate method raises ValueError when area_height is not a positive integer."""
        with pytest.raises(ValueError, match="Area height must be a positive integer. Got 0"):
            TSPGenerator.generate(5, area_height=0, seed=SEED)

    def test_generate_default_parameters(self):
        """Test generate method with default parameters."""
        num_cities = 5
        problem = TSPGenerator.generate(num_cities, seed=SEED)

        assert problem.length == num_cities
        assert problem.coords is not None
        assert problem.distances is not None
        assert problem.source_graph is not None

    def test_generate_custom_parameters(self):
        """Test generate method with custom parameters."""
        num_cities = 5
        problem = TSPGenerator.generate(num_cities, area_width=100, area_height=100, seed=SEED)

        assert problem.length == num_cities
        assert problem.coords is not None
        assert problem.distances is not None
        assert problem.source_graph is not None

    def test_generate_no_duplicate_coordinates(self):
        """Test generate method ensures no duplicate coordinates."""
        problem = TSPGenerator.generate(5, seed=SEED)
        coords = problem.coords

        assert len(coords) == len(set(coords))

    def test_generate_distances(self):
        """Test generate method calculates distances correctly."""
        problem = TSPGenerator.generate(5, seed=SEED)
        distances = problem.distances

        for u, v, d in distances:
            assert d == np.linalg.norm(np.subtract(problem.coords[u], problem.coords[v]))

    def test_generate_graph(self):
        """Test generate method creates a valid graph."""
        num_cities = 5
        problem = TSPGenerator.generate(num_cities, seed=SEED)
        graph = problem.source_graph

        assert graph.number_of_nodes() == num_cities
        assert graph.number_of_edges() == len(problem.distances)

    def test_generate_one_city(self):
        """Test generate method with the minimum number of cities (1)."""
        num_cities = 1
        problem = TSPGenerator.generate(num_cities, seed=SEED)

        assert problem.length == num_cities
        assert len(problem.coords) == num_cities
        assert len(problem.distances) == 0  # No distances when there's only one city
        assert problem.source_graph.number_of_nodes() == num_cities
        assert problem.source_graph.number_of_edges() == 0

    def test_generate_large_number_of_cities(self):
        """Test generate method with a large number of cities."""
        num_cities = 1000
        problem = TSPGenerator.generate(num_cities, seed=SEED)

        assert problem.length == num_cities
        assert len(problem.coords) == num_cities
        assert len(problem.distances) > 0
        assert problem.source_graph.number_of_nodes() == num_cities
        assert problem.source_graph.number_of_edges() == len(problem.distances)

    def test_generate_non_square_area(self):
        """Test generate method with different area width and height."""
        num_cities = 5
        area_width = 300
        area_height = 100
        problem = TSPGenerator.generate(num_cities, area_width=area_width, area_height=area_height, seed=SEED)

        assert problem.length == num_cities
        for x, y in problem.coords:
            assert 0 <= x < area_width
            assert 0 <= y < area_height

    def test_generate_same_seed_reproducibility(self):
        """Test generate method produces the same results when called with the same seed."""
        num_cities = 5
        problem1 = TSPGenerator.generate(num_cities, seed=SEED)
        problem2 = TSPGenerator.generate(num_cities, seed=SEED)

        assert problem1.coords == problem2.coords
        assert problem1.distances == problem2.distances

    def test_generate_randomness_different_seeds(self):
        """Test generate method produces different results with different seeds."""
        num_cities = 5
        problem1 = TSPGenerator.generate(num_cities, seed=45)
        problem2 = TSPGenerator.generate(num_cities, seed=56)

        assert problem1.coords != problem2.coords
        assert problem1.distances != problem2.distances

    def test_generate_negative_area_dimensions(self):
        """Test generate method raises ValueError when area dimensions are negative."""
        with pytest.raises(ValueError, match="Area width must be a positive integer. Got -100"):
            TSPGenerator.generate(5, area_width=-100, seed=SEED)

        with pytest.raises(ValueError, match="Area height must be a positive integer. Got -100"):
            TSPGenerator.generate(5, area_height=-100, seed=SEED)

    def test_get_distances_truncate(self):
        """Test get_distances method truncates distances to integers when truncate=True."""
        coords = [(0, 0), (3, 4), (6, 8)]  # Known distances: 5.0 and 5.0
        expected_distances = [(0, 1, 5), (0, 2, 10), (1, 2, 5)]  # Truncated distances

        distances = TSPGenerator.get_distances(coords)

        assert distances == expected_distances

    def test_get_distances_no_truncate(self):
        """Test get_distances method retains full precision when truncate=False."""
        coords = [(0, 0), (3, 4), (6, 8)]  # Known distances: 5.0 and 10.0
        expected_distances = [(0, 1, 5.0), (0, 2, 10.0), (1, 2, 5.0)]  # Non-truncated distances

        distances = TSPGenerator.get_distances(coords, truncate=False)

        assert distances == expected_distances
