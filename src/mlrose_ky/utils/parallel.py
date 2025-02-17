from joblib import Parallel, delayed
from itertools import product
import time
import mlrose_ky as mlrose
import pandas as pd
import pickle

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        return result, execution_time  # Return result and execution time as a tuple

    return wrapper


def generate_parameter_list(parameter_dict):
    combinations = product(*parameter_dict.values())
    parameters_list = [
        dict(zip(parameter_dict.keys(), combination)) for combination in combinations
    ]

    return parameters_list


@timeit
def rhc_run(
    problem,
    max_attempt,
    max_iter,
    restart,
    seeds,
    state_fitness_call_back=None,
    callback_user_info={},
):
    results = []
    for seed in seeds:
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(
            problem,
            max_attempts=max_attempt,
            max_iters=max_iter,
            restarts=restart,
            init_state=None,
            curve=True,
            random_state=seed,
            state_fitness_callback=state_fitness_call_back,
            callback_user_info=callback_user_info,
        )
        
        results.append({
            "Seed": seed,
            "Best State": best_state,
            "Best Fitness": best_fitness,
            "Fitness Curve": fitness_curve
        })
    
    df = pd.DataFrame(results)
    return df
    # return results


@timeit
def sa_run(
    problem,
    decay,
    max_attempt,
    max_iter,
    seeds,
    state_fitness_call_back=None,
    callback_user_info={},
):
    results = []
    for seed in seeds:
        sa = mlrose.simulated_annealing(
            problem,
            schedule=decay,
            max_attempts=max_attempt,
            max_iters=max_iter,
            init_state=None,
            curve=True,
            random_state=seed,
            state_fitness_callback=state_fitness_call_back,
            callback_user_info=callback_user_info,
        )
        results.append(sa)
    return results


@timeit
def ga_run(
    problem,
    pop_size,
    mutation_prob,
    max_attempt,
    max_iter,
    seeds,
    state_fitness_call_back=None,
    callback_user_info={},
):
    results = []
    for seed in seeds:
        ga = mlrose.genetic_alg(
            problem,
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempt,
            max_iters=max_iter,
            curve=True,
            random_state=seed,
            state_fitness_callback=state_fitness_call_back,
            callback_user_info=callback_user_info,
        )
        results.append(ga)

    return results


@timeit
def mimic_run(
    problem,
    pop_size,
    keep_pct,
    max_attempt,
    max_iter,
    seeds,
    state_fitness_call_back=None,
    callback_user_info={},
):
    results = []
    for seed in seeds:
        mimic = mlrose.mimic(
            problem,
            pop_size=pop_size,
            keep_pct=keep_pct,
            max_attempts=max_attempt,
            max_iters=max_iter,
            curve=True,
            random_state=seed,
            state_fitness_callback=state_fitness_call_back,
            callback_user_info=callback_user_info,
        )
        results.append(mimic)

    return results


def get_results(grid, func, n_jobs=-1, verbose=0):
    params = generate_parameter_list(grid)
    print("Number of params:", len(params))
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(func)(**params) for params in params
    )
    # best_param = get_best_param(results, params)
    return results, params

# example for four peaks
def please_stop(
    iteration,
    state,
    fitness,
    fitness_evaluations,
    user_data,
    attempt=None,
    done=None,
    curve=None,
):
    # Example of a please stop function
    # Hit max attempts or max iters - stop
    if done:
        return False
    if iteration > 1000:
        if fitness >= user_data["target"]:
            return False
        delta = abs(curve[iteration - 1][0] - (curve[-1000:].min(axis=0)[0]))
        if delta < user_data["delta"]:
            return False
    return True


def pickle_dump(filename, results):
    with open(filename, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_read(filename):
    with open(filename, "rb") as handle:
        file_contents = pickle.load(handle)
    return file_contents
