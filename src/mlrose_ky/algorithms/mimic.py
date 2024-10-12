"""Implementation of the MIMIC optimization algorithm for discrete problems."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any

import numpy as np

from mlrose_ky.decorators import short_name


@short_name("mimic")
def mimic(
    problem: Any,
    pop_size: int = 200,
    keep_pct: float = 0.2,
    max_attempts: int = 10,
    noise: float = 0.0,
    max_iters: int | float = np.inf,
    curve: bool = False,
    random_state: int = None,
    state_fitness_callback: Callable = None,
    callback_user_info: dict = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Use MIMIC (Mutual-Information Maximizing Input Clustering) to find the optimum
    for a given optimization problem.

    MIMIC is a probabilistic optimization algorithm that generates new samples
    based on the mutual information between variables.

    Parameters
    ----------
    problem: optimization object
        Object containing the optimization problem to be solved.
        Must be a discrete-state problem, such as `DiscreteOpt()` or `TSPOpt()`.

    pop_size: int, default: 200
        Size of the population to be used in the algorithm.
        Must be a positive integer greater than 0.

    keep_pct: float, default: 0.2
        Proportion of samples to keep at each iteration of the algorithm.
        Must be a float between 0 and 1 (exclusive).

    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
        Must be a positive integer greater than 0.

    noise: float, default: 0.0
        Noise level to be added to the fitness function.
        Must be a float between 0 and 0.1 (inclusive).

    max_iters: int or float, default: np.inf
        Maximum number of iterations of the algorithm.
        Must be a positive integer greater than 0 or `np.inf`.

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
          True if the algorithm is about to terminate (max attempts reached, max iterations reached, or `problem.can_stop()` returns True);
          False otherwise.
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
    - MIMIC cannot be used for solving continuous-state optimization problems.
    - The `state_fitness_callback` function is also called before the optimization loop starts (iteration 0)
      with the initial state and fitness values.
    - MIMIC generates new samples by building a probabilistic model of the best samples and sampling from this model.

    References
    ----------
    De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by Estimating
    Probability Densities. In *Advances in Neural Information Processing Systems*
    (NIPS) 9, pp. 424–430.
    """
    # Validate problem type
    if problem.get_prob_type() == "continuous":
        raise ValueError("MIMIC algorithm cannot be used for continuous optimization problems.")

    # Validate parameters
    if not isinstance(pop_size, int) or pop_size <= 0:
        raise ValueError(f"pop_size must be a positive integer greater than 0. Got {pop_size}")
    if not 0 < keep_pct < 1:
        raise ValueError(f"keep_pct must be between 0 and 1 (exclusive). Got {keep_pct}")
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError(f"max_attempts must be a positive integer greater than 0. Got {max_attempts}")
    if not (isinstance(max_iters, int) or max_iters == np.inf) or max_iters <= 0:
        raise ValueError(f"max_iters must be a positive integer greater than 0 or np.inf. Got {max_iters}")
    if not 0 <= noise <= 0.1:
        raise ValueError(f"noise must be between 0 and 0.1 (inclusive). Got {noise}")
    else:
        problem.noise = noise
    if callback_user_info is not None and not isinstance(callback_user_info, dict):
        raise TypeError(f"callback_user_info must be a dict. Got {type(callback_user_info).__name__}")

    # Set random seed for reproducibility
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize the optimization problem
    fitness_curve = []
    problem.reset()
    problem.random_pop(pop_size)

    # Initial callback invocation (iteration 0)
    if state_fitness_callback is not None:
        if callback_user_info is None:
            callback_user_info = {}
        continue_iterating = state_fitness_callback(
            iteration=0,
            attempt=0,
            done=False,
            state=problem.get_state(),
            fitness=problem.get_adjusted_fitness(),
            fitness_evaluations=problem.fitness_evaluations,
            curve=np.asarray(fitness_curve) if curve else None,
            user_data=callback_user_info,
        )
        if not continue_iterating:
            # Early termination as per callback request
            best_state = problem.get_state().astype(int)
            best_fitness = problem.get_maximize() * problem.get_fitness()
            return best_state, best_fitness, np.asarray(fitness_curve) if curve else None

    # Main optimization loop
    attempts = 0
    iters = 0
    while attempts < max_attempts and iters < max_iters:
        iters += 1
        problem.current_iteration += 1

        # Select the top 'keep_pct' percent of the population
        problem.find_top_pct(keep_pct)

        # Update probability estimates based on the selected samples
        problem.eval_node_probs()

        # Generate a new population using the updated probabilities
        new_sample = problem.sample_pop(pop_size)
        problem.set_population(new_sample)

        # Identify the best state in the new population
        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # Check if the new state is better than the current state
        current_fitness = problem.get_fitness()
        if next_fitness > current_fitness:
            # Improvement found; update state and reset attempts
            problem.set_state(next_state)
            attempts = 0
        else:
            # No improvement; increment attempts
            attempts += 1

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

        # Check if the problem signals to stop
        if problem.can_stop():
            break

    # Prepare the final best state and fitness
    best_state = problem.get_state().astype(int)
    best_fitness = problem.get_maximize() * problem.get_fitness()

    return best_state, best_fitness, np.asarray(fitness_curve) if curve else None
