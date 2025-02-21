# Original implementation by yxlow: https://github.com/freedom89
# Output formatting and enhancements by nkapila6: https://github.com/nkapila6

from joblib import Parallel, delayed
from itertools import product
import time
import mlrose_ky as mlrose
import pandas as pd
import pickle
from pprint import pprint

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        
        if isinstance(result, pd.DataFrame):
            result['Time'] = execution_time
        else:
            print('Warning: Unable to append execution time to the result.')
        # return result, execution_time  # Return result and execution time as a tuple
        return result  # Return result and execution time as a tuple
    return wrapper


def generate_parameter_list(parameter_dict, view_params=False):
    # pprint(parameter_dict)
    combinations = product(*parameter_dict.values())
    parameters_list = [
        dict(zip(parameter_dict.keys(), combination)) for combination in combinations
    ]
    if view_params==True: pprint(parameters_list)
    # pprint(len(parameters_list))
    return parameters_list


@timeit
def rhc_run(
    problem,
    max_attempt,
    max_iter,
    restart,
    seeds,
    param_dict=None,
    state_fitness_call_back=None,
    callback_user_info={}
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
        fitness_value = fitness_curve[:,0].tolist()
        fevals = fitness_curve[:,1].tolist()
        run_result = {
            "Seed": seed,
            "Best State": best_state.tolist(),
            "Best Fitness": best_fitness,
            "Fitness Value": fitness_value,
            "Fevals": fevals,
            'Max Attempt': param_dict['max_attempt'],
            'Max Iters': param_dict['max_iter'],
            'Restart': param_dict['restart'],
            'Problem': param_dict['problem'],
            # "Param dict": param_dict
        }
        # run_result.update(params)
        results.append(run_result)
    
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
    param_dict=None,
    state_fitness_call_back=None,
    callback_user_info={},
):
    results = []
    for seed in seeds:
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
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

        fitness_value = fitness_curve[:,0].tolist()
        fevals = fitness_curve[:,1].tolist()
        run_result = {
            "Seed": seed,
            "Best State": best_state.tolist(),
            "Best Fitness": best_fitness,
            "Fitness Value": fitness_value,
            "Fevals": fevals,
            'Max Attempt': param_dict['max_attempt'],
            'Max Iters': param_dict['max_iter'],
            'Decay': param_dict['decay'],
            'Problem': param_dict['problem'],
            # "Param dict": param_dict
        }
        results.append(run_result)
    
    df = pd.DataFrame(results)
    # results.append(sa)
    return df


@timeit
def ga_run(
    problem,
    pop_size,
    mutation_prob,
    max_attempt,
    max_iter,
    seeds,
    param_dict=None,
    state_fitness_call_back=None,
    callback_user_info={},
):
    results = []
    for seed in seeds:
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(
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
        fitness_value = fitness_curve[:,0].tolist()
        fevals = fitness_curve[:,1].tolist()
        run_result = {
            "Seed": seed,
            "Best State": best_state.tolist(),
            "Best Fitness": best_fitness,
            "Fitness Value": fitness_value,
            "Fevals": fevals,
            'Max Attempt': param_dict['max_attempt'],
            'Max Iters': param_dict['max_iter'],
            'Pop Size': param_dict['pop_size'],
            'Mutation Prob': param_dict['mutation_prob'],
            'Problem': param_dict['problem'],
            # "Param dict": param_dict
        }
        results.append(run_result)
    
    df = pd.DataFrame(results)
    # results.append(sa)
    return df


@timeit
def mimic_run(
    problem,
    pop_size,
    keep_pct,
    max_attempt,
    max_iter,
    seeds,
    param_dict=None,
    state_fitness_call_back=None,
    callback_user_info={},
):
    results = []
    for seed in seeds:
        best_state, best_fitness, fitness_curve = mlrose.mimic(
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
        fitness_value = fitness_curve[:,0].tolist()
        fevals = fitness_curve[:,1].tolist()
        run_result = {
            "Seed": seed,
            "Best State": best_state.tolist(),
            "Best Fitness": best_fitness,
            "Fitness Value": fitness_value,
            "Fevals": fevals,
            'Max Attempt': param_dict['max_attempt'],
            'Max Iters': param_dict['max_iter'],
            'Pop Size': param_dict['pop_size'],
            'Keep Pct': param_dict['keep_pct'],
            'Problem': param_dict['problem'],
            # "Param dict": param_dict
        }
        results.append(run_result)

    df = pd.DataFrame(results)
    # return results
    return df


def get_results(grid, func, n_jobs=-1, verbose=0, view_params=False):
    params = generate_parameter_list(grid, view_params)
    print("Number of params:", len(params))
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(func)(param_dict=params, **params) for params in params
    )
    # best_param = get_best_param(results, params)
    # return results, params
    return results

# Example of custom call back function for Four Peaks.
# def please_stop(
#     iteration,
#     state,
#     fitness,
#     fitness_evaluations,
#     user_data,
#     attempt=None,
#     done=None,
#     curve=None,
# ):
#     # Example of a please stop function
#     # Hit max attempts or max iters - stop
#     if done:
#         return False
#     if iteration > 1000:
#         if fitness >= user_data["target"]:
#             return False
#         delta = abs(curve[iteration - 1][0] - (curve[-1000:].min(axis=0)[0]))
#         if delta < user_data["delta"]:
#             return False
#     return True

# Unnecessary
# def pickle_dump(filename, results):
#     with open(filename, "wb") as handle:
#         pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def pickle_read(filename):
#     with open(filename, "rb") as handle:
#         file_contents = pickle.load(handle)
#     return file_contents
