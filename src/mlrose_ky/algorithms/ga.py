"""Implementation of the Genetic Algorithm optimization algorithm."""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

from typing import Callable, Any, Optional

import numpy as np

from mlrose_ky.decorators import short_name


def _get_hamming_distance_default(population: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """
    Calculate the Hamming distance between a given individual and the rest of the population.

    Parameters
    ----------
    population : np.ndarray
        Population of individuals.
    p1 : np.ndarray
        Individual to compare with the population.

    Returns
    -------
    np.ndarray
        Array of Hamming distances.
    """
    return np.array([np.count_nonzero(p1 != p2) / len(p1) for p2 in population])


def _get_hamming_distance_float(population: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """
    Calculate the Hamming distance (as a float) between a given individual and the rest of the population.

    Parameters
    ----------
    population : np.ndarray
        Population of individuals.
    p1 : np.ndarray
        Individual to compare with the population.

    Returns
    -------
    np.ndarray
        Array of Hamming distances.
    """
    return np.array([np.abs(p1 - p2) / len(p1) for p2 in population])


def _genetic_alg_select_parents(
    pop_size: int,
    problem: Any,
    get_hamming_distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    hamming_factor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select parents for the next generation in the genetic algorithm.

    Parameters
    ----------
    pop_size : int
        Size of the population.
    problem : optimization object
        The optimization problem instance.
    get_hamming_distance_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function to calculate Hamming distance.
    hamming_factor : float, default: 0.0
        Factor to account for Hamming distance in parent selection.

    Returns
    -------
    tuple
        Selected parents (p1, p2) for reproduction.
    """
    mating_probabilities = problem.get_mate_probs()

    if get_hamming_distance_func is not None and hamming_factor > 0.01:
        population = problem.get_population()
        selected = np.random.choice(pop_size, p=mating_probabilities)
        p1 = population[selected]

        hamming_distances = get_hamming_distance_func(population, p1)
        hfa = hamming_factor / (1.0 - hamming_factor)
        hamming_distances = hamming_distances * hfa * mating_probabilities
        hamming_distances /= hamming_distances.sum()

        selected = np.random.choice(pop_size, p=hamming_distances)
        p2 = population[selected]

        return p1, p2

    selected = np.random.choice(pop_size, size=2, p=mating_probabilities)
    p1 = problem.get_population()[selected[0]]
    p2 = problem.get_population()[selected[1]]

    return p1, p2


@short_name("ga")
def genetic_alg(
    problem: Any,
    pop_size: int = 200,
    pop_breed_percent: float = 0.75,
    elite_dreg_ratio: float = 0.99,
    minimum_elites: int = 0,
    minimum_dregs: int = 0,
    mutation_prob: float = 0.1,
    max_attempts: int = 10,
    max_iters: int | float = np.inf,
    curve: bool = False,
    random_state: int = None,
    state_fitness_callback: Callable = None,
    callback_user_info: dict = None,
    hamming_factor: float = 0.0,
    hamming_decay_factor: float = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Use a standard genetic algorithm to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem : optimization object
        Object containing the optimization problem to be solved.
        For example, `DiscreteOpt()` or `ContinuousOpt()`.

    pop_size : int, default: 200
        Size of the population to be used in the genetic algorithm.
        Must be a positive integer greater than 0.

    pop_breed_percent : float, default: 0.75
        Percentage of the population to breed in each iteration.
        Must be a float between 0 and 1 (exclusive).

    elite_dreg_ratio : float, default: 0.99
        Ratio of elites to dregs added directly to the next generation from the current population.
        Must be a float between 0 and 1 (inclusive).
        A higher value favors more elites over dregs.

    minimum_elites : int, default: 0
        Minimum number of elite individuals to be added to the next generation.
        Must be a non-negative integer.

    minimum_dregs : int, default: 0
        Minimum number of dreg (lowest fitness) individuals to be added to the next generation.
        Must be a non-negative integer.

    mutation_prob : float, default: 0.1
        Probability of a mutation at each element of the state vector during reproduction.
        Must be a float between 0 and 1 (inclusive).

    max_attempts : int, default: 10
        Maximum number of attempts to find a better state at each step.
        Must be a positive integer greater than 0.

    max_iters : int or float, default: np.inf
        Maximum number of iterations of the algorithm.
        Must be a positive integer greater than 0 or `np.inf`.

    curve : bool, default: False
        Whether to keep fitness values for a curve.
        If `False`, then no curve is stored.
        If `True`, then a history of fitness values is provided as a third return value.

    random_state : int, default: None
        Seed for the random number generator.

    state_fitness_callback : callable, default: None
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

    hamming_factor : float, default: 0.0
        Factor to account for Hamming distance in parent selection.
        Must be a float between 0 and 1 (inclusive).
        A higher value increases the likelihood of selecting parents with greater diversity.

    hamming_decay_factor : float, default: None
        Decay factor for the `hamming_factor` over iterations.
        If specified, `hamming_factor` is multiplied by this value each iteration.

    Returns
    -------
    best_state : np.ndarray
        Numpy array containing the state that optimizes the fitness function.

    best_fitness : float
        Value of the fitness function at the best state.

    fitness_curve : np.ndarray
        Numpy array of shape (n_iterations, 2), where each row represents:

        - Column 0: Adjusted fitness at the current iteration.
        - Column 1: Cumulative number of fitness evaluations.

        Only returned if the input argument `curve` is `True`.

    Notes
    -----
    - The genetic algorithm evolves a population of candidate solutions by selecting individuals based on fitness, reproducing them with crossover and mutation, and iteratively improving the population.
    - The `state_fitness_callback` function is also called before the optimization loop starts (iteration 0) with the initial state and fitness values.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern Approach*, 3rd edition.
    Prentice Hall, New Jersey, USA.
    """
    # Validate parameters
    if not isinstance(pop_size, int) or pop_size <= 0:
        raise ValueError(f"pop_size must be a positive integer greater than 0. Got {pop_size}")
    if not 0 < pop_breed_percent < 1:
        raise ValueError(f"pop_breed_percent must be between 0 and 1 (exclusive). Got {pop_breed_percent}")
    if not 0 <= elite_dreg_ratio <= 1:
        raise ValueError(f"elite_dreg_ratio must be between 0 and 1 (inclusive). Got {elite_dreg_ratio}")
    if not isinstance(minimum_elites, int) or minimum_elites < 0:
        raise ValueError(f"minimum_elites must be a non-negative integer. Got {minimum_elites}")
    if not isinstance(minimum_dregs, int) or minimum_dregs < 0:
        raise ValueError(f"minimum_dregs must be a non-negative integer. Got {minimum_dregs}")
    if not 0 <= mutation_prob <= 1:
        raise ValueError(f"mutation_prob must be between 0 and 1 (inclusive). Got {mutation_prob}")
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError(f"max_attempts must be a positive integer greater than 0. Got {max_attempts}")
    if not (isinstance(max_iters, int) or max_iters == np.inf) or max_iters <= 0:
        raise ValueError(f"max_iters must be a positive integer greater than 0 or np.inf. Got {max_iters}")
    if not 0 <= hamming_factor <= 1:
        raise ValueError(f"hamming_factor must be between 0 and 1 (inclusive). Got {hamming_factor}")
    if hamming_decay_factor is not None and not 0 <= hamming_decay_factor <= 1:
        raise ValueError(f"hamming_decay_factor must be between 0 and 1 (inclusive). Got {hamming_decay_factor}")

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
            best_state = problem.get_state()
            best_fitness = problem.get_maximize() * problem.get_fitness()
            return best_state, best_fitness, np.asarray(fitness_curve) if curve else None

    # Determine Hamming distance function if needed
    get_hamming_distance_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    if hamming_factor > 0:
        sample_gene = problem.get_population()[0][0]
        if isinstance(sample_gene, float) or sample_gene.dtype == "float64":
            get_hamming_distance_func = _get_hamming_distance_float
        else:
            get_hamming_distance_func = _get_hamming_distance_default

    # Initialize counts for breeding and survivors
    breeding_pop_size = int(pop_size * pop_breed_percent) - (minimum_elites + minimum_dregs)
    survivors_size = pop_size - breeding_pop_size
    dregs_size = max(int(survivors_size * (1.0 - elite_dreg_ratio)) if survivors_size > 1 else 0, minimum_dregs)
    elites_size = max(survivors_size - dregs_size, minimum_elites)

    # Adjust breeding population size if necessary
    if dregs_size + elites_size > survivors_size:
        over_population = dregs_size + elites_size - survivors_size
        breeding_pop_size -= over_population

    attempts = 0
    iters = 0
    while attempts < max_attempts and iters < max_iters:
        iters += 1
        problem.current_iteration += 1

        # Calculate mating probabilities based on fitness
        problem.eval_mate_probs()

        # Create next generation
        next_gen = []
        for _ in range(breeding_pop_size):
            # Select parents
            parent_1, parent_2 = _genetic_alg_select_parents(
                pop_size=pop_size, problem=problem, hamming_factor=hamming_factor, get_hamming_distance_func=get_hamming_distance_func
            )

            # Create offspring through reproduction and mutation
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        # Fill the remaining population with elites and dregs
        if survivors_size > 0:
            last_gen = list(zip(problem.get_population(), problem.get_pop_fitness()))
            sorted_parents = sorted(last_gen, key=lambda f: -f[1])
            best_parents = sorted_parents[:elites_size]
            next_gen.extend([p[0] for p in best_parents])

            if dregs_size > 0:
                worst_parents = sorted_parents[-dregs_size:]
                next_gen.extend([p[0] for p in worst_parents])

        # Ensure the next generation has the correct population size
        next_gen = np.array(next_gen[:pop_size])
        problem.set_population(next_gen)

        # Evaluate the best child in the new generation
        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If the best child is an improvement, update the current state
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
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

        # Decay hamming factor if specified
        if hamming_decay_factor is not None and hamming_factor > 0.0:
            hamming_factor *= hamming_decay_factor
            hamming_factor = max(min(hamming_factor, 1.0), 0.0)

        # Check if the problem signals to stop
        if problem.can_stop():
            break

    # Prepare the final best state and fitness
    best_state = problem.get_state()
    best_fitness = problem.get_maximize() * problem.get_fitness()

    return best_state, best_fitness, np.asarray(fitness_curve) if curve else None
