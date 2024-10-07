"""Unit tests for neural/fitness/"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import re

import numpy as np
import pytest

from mlrose_ky.neural.activation import sigmoid, identity
from mlrose_ky.neural.fitness import NetworkWeights
from tests.globals import sample_data


class TestNeuralFitness:
    """Test cases for the neural.fitness module."""

    def test_invalid_activation_fn(self):
        X = np.array([]).reshape(0, 1)
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = {}

        with pytest.raises(TypeError, match=f"Activation function must be callable, got {type(activation).__name__}"):
            # noinspection PyTypeChecker
            NetworkWeights(X, y, node_list, activation)

    def test_x_empty_y_not_empty(self):
        X = np.array([]).reshape(0, 1)
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        with pytest.raises(ValueError, match="X and y cannot be empty."):
            NetworkWeights(X, y, node_list, activation)

    def test_y_empty_x_not_empty(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([]).reshape(0, 1)
        node_list = [2, 1]
        activation = sigmoid

        with pytest.raises(ValueError, match="X and y cannot be empty"):
            NetworkWeights(X, y, node_list, activation)

    def test_x_y_mismatched_shapes(self):
        X = np.array([[0.1], [0.3], [0.5]])
        y = np.array([[1], [0]])  # Mismatched number of rows
        node_list = [2, 1]
        activation = sigmoid

        with pytest.raises(ValueError, match=re.escape("The length of X (3) and y (2) must be equal.")):
            NetworkWeights(X, y, node_list, activation)

    def test_node_list_too_short(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [1]  # Only one element in node_list
        activation = sigmoid

        with pytest.raises(ValueError, match="node_list must contain at least 2 elements"):
            NetworkWeights(X, y, node_list, activation)

    def test_invalid_is_classifier(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        with pytest.raises(ValueError, match="is_classifier must be True or False"):
            # noinspection PyTypeChecker
            NetworkWeights(X, y, node_list, activation, is_classifier="yes")

    def test_invalid_learning_rate(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        with pytest.raises(ValueError, match="learning_rate must be greater than 0"):
            NetworkWeights(X, y, node_list, activation, learning_rate=0)

    def test_invalid_state_length_in_evaluate(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid
        state = np.array([0.5, 0.5, 0.5])  # Invalid length

        nw = NetworkWeights(X, y, node_list, activation)

        with pytest.raises(ValueError, match="state must have length"):
            nw.evaluate(state)

    def test_initialization_valid(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        nw = NetworkWeights(X, y, node_list, activation)

        assert nw.X.shape == X.shape
        assert nw.y_true.shape == y.shape
        assert nw.node_list == node_list
        assert nw.activation == activation

    def test_initialization_invalid_shapes(self):
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        with pytest.raises(Exception):
            NetworkWeights(X, y, node_list, activation)

    def test_evaluate(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid
        state = np.array([0.5, 0.5])

        nw = NetworkWeights(X, y, node_list, activation)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)

    def test_evaluate_invalid_state_length(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid
        state = np.array([0.5])

        nw = NetworkWeights(X, y, node_list, activation)

        with pytest.raises(Exception):
            nw.evaluate(state)

    def test_get_output_activation(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        nw = NetworkWeights(X, y, node_list, activation)
        output_activation = nw.get_output_activation()

        assert output_activation == sigmoid

    def test_get_prob_type(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid

        nw = NetworkWeights(X, y, node_list, activation)
        prob_type = nw.get_prob_type()

        assert prob_type == "continuous"

    def test_no_hidden_layers(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = identity
        state = np.array([0.5, 0.5])

        nw = NetworkWeights(X, y, node_list, activation)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)

    def test_multiple_hidden_layers(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 3, 2, 1]
        activation = sigmoid
        state = np.array([0.5] * 14)

        nw = NetworkWeights(X, y, node_list, activation)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)
        updates = nw.calculate_updates()
        assert isinstance(updates, list)
        assert len(updates) == len(node_list) - 1

    def test_multiclass_classification(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1, 0, 0], [0, 1, 0]])
        node_list = [2, 3]
        activation = sigmoid
        state = np.array([0.5] * 6)

        nw = NetworkWeights(X, y, node_list, activation)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)
        assert nw.output_activation.__name__ == "softmax"

    def test_regression(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1.0], [0.5]])
        node_list = [2, 1]
        activation = identity
        state = np.array([0.5, 0.5])

        nw = NetworkWeights(X, y, node_list, activation, is_classifier=False)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)
        assert nw.output_activation.__name__ == "identity"
        assert nw.loss.__name__ == "mean_squared_error"

    def test_invalid_activation_function(self):
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]  # Adjusted to have 2 input nodes

        def invalid_activation(x):
            """An invalid activation function that doesn't have the required signature"""
            return x

        with pytest.raises(TypeError):
            NetworkWeights(X, y, node_list, invalid_activation)

    def test_extreme_values(self):
        X = np.array([[1e10], [1e-10]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid
        state = np.array([0.5, 0.5])

        nw = NetworkWeights(X, y, node_list, activation)
        fitness = nw.evaluate(state)

        assert isinstance(fitness, float)

    def test_empty_dataset(self):
        X = np.array([]).reshape(0, 1)
        y = np.array([]).reshape(0, 1)
        node_list = [2, 1]
        activation = sigmoid

        with pytest.raises(ValueError, match="X and y cannot be empty"):
            NetworkWeights(X, y, node_list, activation)

    def test_evaluate_no_bias_classifier(self, sample_data):
        """Test evaluation of network weights without bias for classification."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_classifier, node_list, activation=identity, bias=bias)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.7393

    def test_evaluate_no_bias_multi(self, sample_data):
        """Test evaluation of network weights without bias for multiclass classification."""
        X, _, y_multiclass, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2]  # Unsure why last layer needs 2 nodes even though bias is False
        fitness = NetworkWeights(X, y_multiclass, node_list, activation=identity, bias=bias)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(4) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.7183

    def test_evaluate_no_bias_regressor(self, sample_data):
        """Test evaluation of network weights without bias for regression."""
        X, _, _, y_regressor = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_regressor, node_list, activation=identity, bias=bias, is_classifier=False)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.5542

    def test_evaluate_bias_regressor(self, sample_data):
        """Test evaluation of network weights with bias for regression."""
        X, _, _, y_regressor = sample_data
        hidden_nodes = [2]
        bias = True
        node_list = [5, *hidden_nodes, 1]  # Unsure why this first number needs to be 5 even though X.shape[1] is 6
        fitness = NetworkWeights(X, y_regressor, node_list, bias=bias, activation=identity, is_classifier=False)

        a = list(np.arange(10) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b

        assert round(fitness.evaluate(np.asarray(weights)), 4) == 0.4363

    def test_calculate_updates(self, sample_data):
        """Test calculation of weight updates for the network."""
        X, y_classifier, _, _ = sample_data
        hidden_nodes = [2]
        bias = False
        node_list = [X.shape[1], *hidden_nodes, 2 if bias else 1]
        fitness = NetworkWeights(X, y_classifier, node_list, activation=identity, bias=bias, is_classifier=False, learning_rate=1)

        a = list(np.arange(8) + 1)
        b = list(0.01 * (np.arange(2) + 1))
        weights = a + b
        fitness.evaluate(np.asarray(weights))

        updates = list(fitness.calculate_updates())
        update1 = np.array([[-0.0017, -0.0034], [-0.0046, -0.0092], [-0.0052, -0.0104], [0.0014, 0.0028]])
        update2 = np.array([[-3.17], [-4.18]])

        assert np.allclose(updates[0], update1, atol=0.001) and np.allclose(updates[1], update2, atol=0.001)

    def test_y_1d_reshaped(self):
        """Test that a 1D y array is reshaped correctly."""
        X = np.array([[0.1], [0.3]])
        y = np.array([1, 0])  # 1D array
        node_list = [2, 1]
        activation = sigmoid

        nw = NetworkWeights(X, y, node_list, activation)

        assert nw.y_true.shape == (2, 1), "y should be reshaped to 2D array with shape (2, 1)"

    def test_y_wrong_num_columns(self):
        """Test that ValueError is raised when y has incorrect number of columns."""
        X = np.array([[0.1], [0.3]])
        y = np.array([[1, 0], [0, 1]])  # 2 columns
        node_list = [2, 1]  # Last element is 1, expecting y to have 1 column
        activation = sigmoid

        with pytest.raises(ValueError, match="The number of columns in y must equal 1."):
            NetworkWeights(X, y, node_list, activation)

    def test_invalid_bias(self):
        """Test that ValueError is raised when bias is not a boolean."""
        X = np.array([[0.1], [0.3]])
        y = np.array([[1], [0]])
        node_list = [2, 1]
        activation = sigmoid
        bias = "yes"  # Invalid type

        with pytest.raises(TypeError, match=re.escape("bias must be a bool (True or False).")):
            # noinspection PyTypeChecker
            NetworkWeights(X, y, node_list, activation, bias=bias)

    def test_initialization_invalid_shapes_specific(self):
        """Test initialization with invalid X and y shapes."""
        X = np.array([[0.1, 0.2], [0.3, 0.4]])  # 2 columns
        y = np.array([[1], [0], [1]])  # 3 rows
        node_list = [2, 1]
        activation = sigmoid

        with pytest.raises(ValueError, match=re.escape("The length of X (2) and y (3) must be equal.")):
            NetworkWeights(X, y, node_list, activation)
