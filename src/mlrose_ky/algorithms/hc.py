"""Implementation of the Hill Climbing optimization algorithm."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any

import numpy as np

from mlrose_ky.decorators import short_name


@short_name("hc")
def hill_climb(
    problem: Any,
    max_iters: int | float = np.inf,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: int = None,
    state_fitness_callback: Callable = None,
    callback_user_info: Any = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Use standard hill climbing to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing the optimization problem to be solved.
        For example, `DiscreteOpt()`, `ContinuousOpt()`, or `TSPOpt()`.
    max_iters: int or float, default: np.inf
        Maximum number of iterations before the algorithm terminates.
        Must be a positive integer greater than 0, or `np.inf`.
    init_state: np.ndarray, default: None
        1-D Numpy array containing the starting state for the algorithm.
        If `None`, then a random state is used.
    curve: bool, default: False
        Whether to keep fitness values for a curve.
        If `False`, then no curve is stored.
        If `True`, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        Seed for the random number generator.
    state_fitness_callback: Callable, default: None
        If specified, this callback function is invoked once per iteration with
        the following parameters:

        - iteration: int
        - attempt: None
        - done: bool
        - state: np.ndarray
        - fitness: float
        - curve: np.ndarray or None
        - user_data: any

        The callback should return a boolean: `True` to continue iterating,
        or `False` to stop.
    callback_user_info: any, default: None
        User data passed as the last parameter of the callback function.

    Returns
    -------
    best_state: np.ndarray
        Numpy array containing the state that optimizes the fitness function.
    best_fitness: float
        Value of the fitness function at the best state.
    fitness_curve: np.ndarray
        Numpy array of shape (n_iterations, 2), where each row contains:

        - Adjusted fitness at the current iteration.
        - Cumulative number of fitness evaluations.

        Only returned if the input argument `curve` is `True`.

    Notes
    -----
    The `state_fitness_callback` function is also called before the optimization
    loop starts (iteration 0) with the initial state and fitness values.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern Approach*, 3rd edition.
    Prentice Hall, New Jersey, USA.
    """
    if not (isinstance(max_iters, int) or max_iters == np.inf or max_iters.is_integer()) or max_iters < 0:
        raise ValueError(f"max_iters must be a positive integer. Got {max_iters}")
    if init_state is not None and len(init_state) != problem.get_length():
        raise ValueError(f"init_state must have the same length as the problem. Expected {problem.get_length()}, got {len(init_state)}")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -np.inf
    best_state = None

    fitness_curve = []
    best_fitness_curve = []

    # Initialize optimization problem
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    # Initial callback invocation
    if state_fitness_callback is not None:
        continue_iterating = state_fitness_callback(
            iteration=0,
            attempt=None,
            done=False,
            state=problem.get_state(),
            fitness=problem.get_adjusted_fitness(),
            curve=np.asarray(fitness_curve) if curve else None,
            user_data=callback_user_info,
        )
        if not continue_iterating:
            return problem.get_state(), best_fitness, np.asarray(best_fitness_curve) if curve else None

    # Main optimization loop
    iters = 0
    while iters < max_iters:
        iters += 1
        problem.current_iteration += 1

        # Find neighbors and determine best neighbor
        problem.find_neighbors()
        next_state = problem.best_neighbor()
        next_fitness = problem.eval_fitness(next_state)

        # If curve is True, append current fitness and evaluations to fitness_curve
        if curve:
            fitness_curve.append((problem.get_adjusted_fitness(), problem.fitness_evaluations))

        # Invoke callback
        if state_fitness_callback is not None:
            max_attempts_reached = (iters == max_iters) or problem.can_stop()
            continue_iterating = state_fitness_callback(
                iteration=iters,
                attempt=None,
                done=max_attempts_reached,
                state=problem.get_state(),
                fitness=problem.get_adjusted_fitness(),
                curve=np.asarray(fitness_curve) if curve else None,
                user_data=callback_user_info,
            )
            # Break out if requested
            if not continue_iterating:
                break

        # If best neighbor is an improvement, move to that state
        current_fitness = problem.get_fitness()
        if next_fitness > current_fitness:
            problem.set_state(next_state)
        else:
            break

    # Update best state and best fitness
    if problem.get_fitness() > best_fitness:
        best_fitness = problem.get_fitness()
        best_state = problem.get_state()
        if curve:
            best_fitness_curve = [*fitness_curve]

    best_fitness = problem.get_maximize() * best_fitness

    return best_state, best_fitness, np.asarray(best_fitness_curve) if curve else None
