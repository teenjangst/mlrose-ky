"""Implementation of the Simulated Annealing optimization algorithm."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any

import numpy as np

from mlrose_ky.algorithms.decay import GeomDecay
from mlrose_ky.decorators import short_name


@short_name("sa")
def simulated_annealing(
    problem: Any,
    schedule: Any = GeomDecay(),
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    init_state: np.ndarray = None,
    curve: bool = False,
    random_state: int = None,
    state_fitness_callback: Callable = None,
    callback_user_info: Any = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Use simulated annealing to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing the optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()`, or
        :code:`TSPOpt()`.
    schedule: schedule object, default: :code:`mlrose_ky.GeomDecay()`
        Schedule used to determine the value of the temperature parameter.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int or float, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: np.ndarray, default: None
        1-D Numpy array containing the starting state for the algorithm.
        If :code:`None`, then a random state is used.
    curve: bool, default: False
        Whether to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        Seed for the random number generator.
    state_fitness_callback: callable, default: None
        If specified, this callback function is invoked once per iteration with
        the following parameters:

        - iteration: int
        - attempt: int
        - done: bool
        - state: np.ndarray
        - fitness: float
        - fitness_evaluations: int
        - curve: np.ndarray or None
        - user_data: any

        The callback should return a boolean: :code:`True` to continue iterating,
        or :code:`False` to stop.

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

        Only returned if the input argument :code:`curve` is :code:`True`.

    Notes
    -----
    The `state_fitness_callback` function is also called before the optimization
    loop starts (iteration 0) with the initial state and fitness values.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern Approach*, 3rd edition.
    Prentice Hall, New Jersey, USA.
    """
    if not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError(f"max_attempts must be a positive integer. Got {max_attempts}")
    if not (isinstance(max_iters, int) or max_iters == np.inf) or max_iters < 0:
        raise ValueError(f"max_iters must be a positive integer or np.inf. Got {max_iters}")
    if init_state is not None and len(init_state) != problem.get_length():
        raise ValueError(
            f"init_state must have the same length as the problem. Expected length {problem.get_length()}, got {len(init_state)}"
        )

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize problem
    fitness_curve = []
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)
    if state_fitness_callback is not None:
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

    # Main optimization loop
    attempts = 0
    iters = 0
    continue_iterating = True
    while attempts < max_attempts and iters < max_iters:
        # Evaluate the temperature at the current iteration
        temp = schedule.evaluate(iters)
        iters += 1
        problem.current_iteration += 1

        if temp == 0:
            break  # Terminate if temperature is zero
        else:
            # Find a random neighbor and evaluate its fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # Calculate the change in fitness and acceptance probability
            current_fitness = problem.get_fitness()
            delta_e = next_fitness - current_fitness
            prob = np.exp(delta_e / temp)

            # Decide whether to accept the new state
            if delta_e > 0 or np.random.uniform() < prob:
                problem.set_state(next_state)
                attempts = 0  # Reset attempts since an improvement was found
            else:
                attempts += 1  # Increment attempts since no improvement

        if curve:
            fitness_curve.append((problem.get_adjusted_fitness(), problem.fitness_evaluations))

        # invoke callback
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

        # break out if requested
        if not continue_iterating:
            break

    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state()

    return best_state, best_fitness, np.asarray(fitness_curve) if curve else None
