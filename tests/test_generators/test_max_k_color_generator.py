"""Unit tests for generators/"""

import networkx as nx
import pytest

from tests.globals import SEED

from mlrose_ky.generators import MaxKColorGenerator


# noinspection PyTypeChecker
class TestMaxKColorGenerator:

    def test_generate_negative_max_colors(self):
        """Test generate method raises ValueError when max_colors is a negative integer."""
        with pytest.raises(ValueError, match="Max colors must be a positive integer or None. Got -3"):
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=10, max_colors=-3)

    def test_generate_non_integer_max_colors(self):
        """Test generate method raises ValueError when max_colors is a non-integer value."""
        with pytest.raises(ValueError, match="Max colors must be a positive integer or None. Got five"):
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=10, max_connections_per_node=3, max_colors="five")

    def test_generate_seed_float(self):
        """Test generate method raises ValueError when SEED is a float."""
        with pytest.raises(ValueError, match="Seed must be an integer. Got 1.5"):
            MaxKColorGenerator.generate(seed=1.5)

    def test_generate_float_number_of_nodes(self):
        """Test generate method raises ValueError when number_of_nodes is a float."""
        with pytest.raises(ValueError, match="Number of nodes must be a positive integer. Got 10.5"):
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=10.5)

    def test_generate_max_connections_per_node_float(self):
        """Test generate method raises ValueError when max_connections_per_node is a float."""
        with pytest.raises(ValueError, match="Max connections per node must be a positive integer. Got 4.5"):
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=10, max_connections_per_node=4.5)

    def test_generate_maximize_string(self):
        """Test generate method raises ValueError when maximize is a string."""
        with pytest.raises(ValueError, match="Maximize must be a boolean. Got true"):
            MaxKColorGenerator.generate(seed=SEED, maximize="true")

    def test_generate_zero_nodes(self):
        """Test generate method raises ValueError when number_of_nodes is zero."""
        with pytest.raises(ValueError, match="Number of nodes must be a positive integer. Got 0"):
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=0)

    def test_generate_no_edges(self):
        """Test generate method with no possible connections."""
        with pytest.raises(ValueError, match="Max connections per node must be a positive integer. Got 0"):
            MaxKColorGenerator.generate(seed=SEED, number_of_nodes=10, max_connections_per_node=0)

    def test_generate_default_parameters(self):
        """Test generate method with default parameters."""
        problem = MaxKColorGenerator.generate(seed=SEED)

        assert problem.length == 20
        assert problem.source_graph.number_of_edges() > 0

    def test_generate_maximum_colors(self):
        """Test generate method with maximum number of colors."""
        number_of_nodes = 5
        max_colors = 100
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_colors=max_colors)

        assert problem.max_val == max_colors

    def test_generate_single_node_one_connection(self):
        """Test generate method with one node and up to one connection."""
        number_of_nodes = 1
        max_connections_per_node = 1
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() == 0

    def test_generate_single_node_two_connections(self):
        """Test generate method with one node and up to two connections."""
        number_of_nodes = 1
        max_connections_per_node = 2
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() == 0

    def test_generate_two_nodes_one_connection(self):
        """Test generate method with two nodes and up to one connection."""
        number_of_nodes = 2
        max_connections_per_node = 1
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() == 1

    def test_generate_two_nodes_two_connections(self):
        """Test generate method with two nodes and up to two connections."""
        number_of_nodes = 2
        max_connections_per_node = 2
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() == 1

    def test_generate_large_graph(self):
        """Test generate method with a large graph."""
        number_of_nodes = 100
        max_connections_per_node = 10
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() > 0

    def test_generate_max_colors_none(self):
        """Test generate method with max_colors set to None."""
        number_of_nodes = 5
        max_connections_per_node = 3
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.max_val > 1

    def test_generate_large_max_colors(self):
        """Test generate method with a large max_colors value."""
        number_of_nodes = 10
        max_connections_per_node = 3
        max_colors = 100
        problem = MaxKColorGenerator.generate(
            SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node, max_colors=max_colors
        )

        assert problem.length == number_of_nodes
        assert problem.max_val == max_colors

    def test_generate_large_max_connections(self):
        """Test generate method with large max_connections_per_node value."""
        number_of_nodes = 10
        max_connections_per_node = 9
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        assert problem.length == number_of_nodes
        assert problem.source_graph.number_of_edges() > 0

    def test_generate_unreachable_nodes(self):
        """Test generate method adds edges to ensure graph connectivity when some nodes are unreachable."""
        number_of_nodes = 4
        max_connections_per_node = 1  # Low connections to increase likelihood of disconnected components
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        # Check that all nodes are reachable from any other node
        for node in range(number_of_nodes):
            assert len(nx.bfs_tree(problem.source_graph, node).nodes) == number_of_nodes

    def test_generate_fully_connected_graph(self):
        """Test generate method creates a fully connected graph when max_connections_per_node is large enough."""
        number_of_nodes = 4
        max_connections_per_node = number_of_nodes - 1  # Enough to potentially fully connect the graph
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        # Check that all nodes are reachable from any other node
        for node in range(number_of_nodes):
            assert len(nx.bfs_tree(problem.source_graph, node).nodes) == number_of_nodes
        assert problem.source_graph.number_of_edges() == number_of_nodes * (number_of_nodes - 1) / 2  # Full connectivity

    def test_generate_single_disconnected_node(self):
        """Test generate method adds an edge to connect a single disconnected node."""
        number_of_nodes = 3
        max_connections_per_node = 1  # Force a disconnected node scenario
        problem = MaxKColorGenerator.generate(seed=SEED, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        # Check that all nodes are reachable from any other node
        for node in range(number_of_nodes):
            assert len(nx.bfs_tree(problem.source_graph, node).nodes) == number_of_nodes

    def test_generate_unreachable_nodes_coverage(self):
        """Test generate method adds edges to ensure connectivity when some nodes are initially unreachable."""
        number_of_nodes = 6
        max_connections_per_node = 1  # Low connections to increase likelihood of disconnected components

        # Use a seed that leads to unreachable nodes
        problem = MaxKColorGenerator.generate(seed=100, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node)

        # Check that all nodes are reachable from any other node
        for node in range(number_of_nodes):
            assert len(nx.bfs_tree(problem.source_graph, node).nodes) == number_of_nodes
