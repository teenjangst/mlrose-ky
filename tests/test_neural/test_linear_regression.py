"""Unit tests for neural/"""

# Author: Kyle Nakamura
# License: BSD 3-clause

import numpy as np

from tests.globals import sample_data
from mlrose_ky import identity
from mlrose_ky.neural.linear_regression import LinearRegression


class TestLinearRegression:
    """Test cases for the LinearRegression class."""

    def test_fit_random_hill_climb(self, sample_data):
        """Test fitting LinearRegression using random hill climb."""
        X, y_classifier, _, _ = sample_data
        network = LinearRegression(bias=False, learning_rate=1, clip_max=1)

        num_weights = X.shape[1]
        weights = np.ones(num_weights)

        network.fit(X, y_classifier, init_weights=weights)
        fitted = network.fitted_weights

        assert sum(fitted) <= num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_fit_simulated_annealing(self, sample_data):
        """Test fitting LinearRegression using simulated annealing."""
        X, y_classifier, _, _ = sample_data
        network = LinearRegression(algorithm="simulated_annealing", bias=False, learning_rate=1, clip_max=1)

        num_weights = X.shape[1]
        weights = np.ones(num_weights)

        network.fit(X, y_classifier, init_weights=weights)
        fitted = network.fitted_weights

        assert sum(fitted) <= num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_fit_genetic_alg(self, sample_data):
        """Test fitting LinearRegression using genetic algorithm."""
        X, y_classifier, _, _ = sample_data
        network = LinearRegression(algorithm="genetic_alg", bias=False, learning_rate=1, clip_max=1, max_attempts=1)

        num_weights = X.shape[1]
        weights = np.ones(num_weights)

        network.fit(X, y_classifier, init_weights=weights)
        fitted = network.fitted_weights

        assert sum(fitted) <= num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_fit_gradient_descent(self, sample_data):
        """Test fitting LinearRegression using gradient descent."""
        X, y_classifier, _, _ = sample_data
        network = LinearRegression(algorithm="gradient_descent", bias=False, clip_max=1)

        num_weights = X.shape[1]
        weights = np.ones(num_weights)

        network.fit(X, y_classifier, init_weights=weights)
        fitted = network.fitted_weights

        assert sum(fitted) <= num_weights and len(fitted) == num_weights and min(fitted) >= -1 and max(fitted) <= 1

    def test_predict_no_bias(self, sample_data):
        """Test prediction without bias in LinearRegression."""
        X, _, _, _ = sample_data
        bias = False
        network = LinearRegression(bias=bias, learning_rate=1, clip_max=1)

        first_layer_size = X.shape[1] + (1 if bias else 0)

        network.fitted_weights = np.ones(first_layer_size)
        network.node_list = [first_layer_size, 1]
        network.output_activation = identity

        x = np.reshape(np.array([2, 0, 4, 4, 2, 1]), [6, 1])
        assert np.array_equal(network.predict(X), x)

    def test_predict_bias(self, sample_data):
        """Test prediction with bias in LinearRegression."""
        X, _, _, _ = sample_data
        bias = True
        network = LinearRegression(bias=bias, learning_rate=1, clip_max=1)

        first_layer_size = X.shape[1] + (1 if bias else 0)

        network.fitted_weights = np.ones(first_layer_size)
        network.node_list = [first_layer_size, 1]
        network.output_activation = identity

        x = np.reshape(np.array([3, 1, 5, 5, 3, 2]), [6, 1])
        assert np.array_equal(network.predict(X), x)
