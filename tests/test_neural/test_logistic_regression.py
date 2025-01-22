"""Unit tests for neural/"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import numpy as np

from tests.globals import sample_data
from mlrose_ky import sigmoid
from mlrose_ky.neural.logistic_regression import LogisticRegression


class TestLogisticRegression:
    """Test cases for the LogisticRegression class."""

    def test_fit_random_hill_climb(self, sample_data):
        """Test fitting LogisticRegression using random hill climb."""
        X, y_classifier, _, _ = sample_data
        bias = False
        network = LogisticRegression(bias=bias, learning_rate=1, clip_max=1)

        num_weights = X.shape[1] + (1 if bias else 0)
        weights = np.ones(num_weights)
        network.fit(X, y_classifier, init_weights=weights)
        fitted = network.fitted_weights

        assert sum(fitted) <= num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_fit_simulated_annealing(self, sample_data):
        """Test fitting LogisticRegression using simulated annealing."""
        X, y_classifier, _, _ = sample_data
        bias = True
        network = LogisticRegression(algorithm="simulated_annealing", bias=bias, learning_rate=1, clip_max=1)

        num_weights = X.shape[1] + (1 if bias else 0)
        weights = np.ones(num_weights)
        network.fit(X, y_classifier, init_weights=weights)
        fitted = network.fitted_weights

        assert sum(fitted) <= num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_fit_genetic_alg(self, sample_data):
        """Test fitting LogisticRegression using genetic algorithm."""
        X, y_classifier, _, _ = sample_data
        bias = False
        network = LogisticRegression(algorithm="genetic_alg", bias=bias, learning_rate=1, clip_max=1, max_iters=2)

        num_weights = X.shape[1] + (1 if bias else 0)
        network.fit(X, y_classifier)
        fitted = network.fitted_weights

        assert sum(fitted) <= num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_fit_gradient_descent(self, sample_data):
        """Test fitting LogisticRegression using gradient descent."""
        X, y_classifier, _, _ = sample_data
        bias = False
        network = LogisticRegression(algorithm="gradient_descent", bias=bias, clip_max=1)

        num_weights = X.shape[1] + (1 if bias else 0)
        weights = np.ones(num_weights)
        network.fit(X, y_classifier, init_weights=weights)
        fitted = network.fitted_weights

        assert sum(fitted) <= num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_predict_no_bias(self, sample_data):
        """Test prediction without bias in LogisticRegression."""
        X, _, _, _ = sample_data
        bias = False
        network = LogisticRegression(bias=bias, learning_rate=1, clip_max=1)

        first_layer_size = X.shape[1] + (1 if bias else 0)
        node_list = [first_layer_size, 1]
        network.fitted_weights = np.array([-1, 1, 1, 1])
        network.node_list = node_list
        network.output_activation = sigmoid

        probs = np.reshape(np.array([0.88080, 0.5, 0.88080, 0.88080, 0.88080, 0.26894]), [6, 1])
        labels = np.reshape(np.array([1, 0, 1, 1, 1, 0]), [6, 1])

        assert np.array_equal(network.predict(X), labels) and np.allclose(network.predicted_probs, probs, atol=0.0001)

    def test_predict_bias(self, sample_data):
        """Test prediction with bias in LogisticRegression."""
        X, _, _, _ = sample_data
        bias = True
        network = LogisticRegression(bias=bias, learning_rate=1, clip_max=1)

        first_layer_size = X.shape[1] + (1 if bias else 0)
        node_list = [first_layer_size, 1]
        network.fitted_weights = np.array([-1, 1, 1, 1, -1])
        network.node_list = node_list
        network.output_activation = sigmoid

        probs = np.reshape(np.array([0.73106, 0.26894, 0.73106, 0.73106, 0.73106, 0.11920]), [6, 1])
        labels = np.reshape(np.array([1, 0, 1, 1, 1, 0]), [6, 1])

        assert np.array_equal(network.predict(X), labels) and np.allclose(network.predicted_probs, probs, atol=0.0001)
