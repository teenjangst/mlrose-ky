"""Classes for defining neural network weight optimization problems."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any

import numpy as np

from mlrose_ky.decorators import short_name
from mlrose_ky.neural.utils import flatten_weights


@short_name("gd")
def gradient_descent(
    problem: Any,
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: int = None,
    state_fitness_callback: Callable = None,
    callback_user_info: dict = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Use gradient descent to find the optimal weights for a neural network.

    Parameters
    ----------
    problem: optimization object
        Object containing the optimization problem to be solved.
        Must be an instance of a neural network weight optimization problem.

    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
        Must be a positive integer greater than 0.

    max_iters: int or float, default: np.inf
        Maximum number of iterations of the algorithm.
        Must be a positive integer greater than 0 or `np.inf`.

    init_state: np.ndarray, default: None
        Numpy array containing the starting state for the algorithm.
        If `None`, then a random state is used.

    curve: bool, default: False
        Whether to keep fitness values for a curve.
        If `False`, then no curve is stored.
        If `True`, then a history of fitness values is provided as a third return value.

    random_state: int, default: None
        Seed for the random number generator.

    state_fitness_callback: callable, default: None
        If specified, this callback function is invoked once per iteration with the following parameters:

        - iteration: int
          The current iteration number (starting from 0). `iteration=0` indicates the initial state before the optimization loop starts.
        - attempt: int
          The current number of consecutive unsuccessful attempts to find a better state.
        - done: bool
          True if the algorithm is about to terminate (max attempts reached, max iterations reached, or `problem.can_stop()` returns True); False otherwise.
        - state: np.ndarray
          The current state vector.
        - fitness: float
          The current adjusted fitness value.
        - fitness_evaluations: int
          The cumulative number of fitness evaluations.
        - curve: np.ndarray or None
          The fitness curve up to the current iteration, or `None` if `curve=False`.
        - user_data: dict
          The user data passed in `callback_user_info`.

        The callback should return a boolean: `True` to continue iterating, or `False` to stop.

    callback_user_info: dict, default: None
        Dictionary of user-managed data passed as the `user_data` parameter of the callback function.

    Returns
    -------
    best_state: np.ndarray
        Numpy array containing the state that optimizes the fitness function.

    best_fitness: float
        Value of the fitness function at the best state.

    fitness_curve: np.ndarray
        Numpy array of shape (n_iterations, 2), where each row represents:

        - Column 0: Adjusted fitness at the current iteration.
        - Column 1: Cumulative number of fitness evaluations.

        Only returned if the input argument `curve` is `True`.

    Notes
    -----
    - The `gradient_descent` function is specifically designed for optimizing neural network weights.
    - The `state_fitness_callback` function is also called before the optimization loop starts (iteration 0) with the initial state and fitness values.

    """
    # Validate parameters
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError(f"max_attempts must be a positive integer greater than 0. Got {max_attempts}")
    if not (isinstance(max_iters, int) or max_iters == np.inf) or max_iters <= 0:
        raise ValueError(f"max_iters must be a positive integer greater than 0 or np.inf. Got {max_iters}")
    if callback_user_info is not None and not isinstance(callback_user_info, dict):
        raise TypeError(f"callback_user_info must be a dict. Got {type(callback_user_info).__name__}")

    # Set random seed for reproducibility
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize the optimization problem
    fitness_curve = []
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    # Initial callback invocation (iteration 0)
    if state_fitness_callback is not None:
        if callback_user_info is None:
            callback_user_info = {}
        state_fitness_callback(
            iteration=0,
            attempt=0,
            done=False,
            state=problem.get_state(),
            fitness=problem.get_adjusted_fitness(),
            fitness_evaluations=problem.fitness_evaluations,
            curve=np.asarray(fitness_curve) if curve else None,
            user_data=callback_user_info,
        )

    # Initialize best state and best fitness
    best_fitness = problem.get_fitness()
    best_state = problem.get_state()

    attempts = 0
    iters = 0
    while attempts < max_attempts and iters < max_iters:
        iters += 1

        # Calculate the gradient updates
        updates = flatten_weights(problem.calculate_updates())

        # Update the state (weights) using the calculated gradients
        next_state = problem.update_state(updates)
        next_fitness = problem.eval_fitness(next_state)

        current_fitness = problem.get_fitness()
        # Check if the new state is better
        if problem.get_maximize() * next_fitness > problem.get_maximize() * current_fitness:
            attempts = 0  # Reset attempts since improvement was found
        else:
            attempts += 1  # Increment attempts since no improvement

        # Record fitness curve if requested
        if curve:
            fitness_curve.append((problem.get_adjusted_fitness(), problem.fitness_evaluations))

        # Invoke callback function
        if state_fitness_callback is not None:
            max_attempts_reached = attempts == max_attempts or iters == max_iters or problem.can_stop()
            continue_iterating = state_fitness_callback(
                iteration=iters,
                attempt=attempts,
                done=max_attempts_reached,
                state=problem.get_state(),
                fitness=problem.get_adjusted_fitness(),
                fitness_evaluations=problem.fitness_evaluations,
                curve=np.asarray(fitness_curve) if curve else None,
                user_data=callback_user_info,
            )
            # Break out if callback requests termination
            if not continue_iterating:
                break

        # Update best state and best fitness if current is better
        if problem.get_maximize() * next_fitness > problem.get_maximize() * best_fitness:
            best_fitness = next_fitness
            best_state = next_state

        # Update the problem's current state
        problem.set_state(next_state)

        # Check if the problem signals to stop
        if problem.can_stop():
            break

    return best_state, best_fitness, np.asarray(fitness_curve) if curve else None
