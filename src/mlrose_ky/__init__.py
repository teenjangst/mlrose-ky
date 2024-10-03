"""
Initialization file for mlrose-ky.

This module exposes all sub-module imports to the module-level, making it easier to use the package's functionalities.
"""

# Authors: Genevieve Hayes (modified by Andrew Rollings, Kyle Nakamura)
# License: BSD 3-clause

# noinspection PyUnresolvedReferences
from .fitness import ContinuousPeaks, CustomFitness, FlipFlop, FourPeaks, Knapsack, MaxKColor, OneMax, Queens, SixPeaks, TravellingSales

# noinspection PyUnresolvedReferences
from .gridsearch import GridSearchMixin

# noinspection PyUnresolvedReferences
from .opt_probs import ContinuousOpt, DiscreteOpt, FlipFlopOpt, KnapsackOpt, MaxKColorOpt, QueensOpt, TSPOpt

# noinspection PyUnresolvedReferences
from .runners import GARunner, MIMICRunner, NNGSRunner, RHCRunner, SARunner, SKMLPRunner, build_data_filename

# noinspection PyUnresolvedReferences
from .samples import plot_synthetic_dataset, SyntheticData

# noinspection PyUnresolvedReferences
from .algorithms import (
    ArithDecay,
    ChangeOneMutator,
    CustomSchedule,
    DiscreteMutator,
    ExpDecay,
    GeomDecay,
    OnePointCrossOver,
    ShiftOneMutator,
    SwapMutator,
    TSPCrossOver,
    UniformCrossOver,
    genetic_alg,
    gradient_descent,
    hill_climb,
    mimic,
    random_hill_climb,
    simulated_annealing,
)

# noinspection PyUnresolvedReferences
from .generators import (
    ContinuousPeaksGenerator,
    FlipFlopGenerator,
    FourPeaksGenerator,
    KnapsackGenerator,
    MaxKColorGenerator,
    OneMaxGenerator,
    QueensGenerator,
    SixPeaksGenerator,
    TSPGenerator,
)

# noinspection PyUnresolvedReferences,PyProtectedMember
from .neural import (
    LinearRegression,
    LogisticRegression,
    NNClassifier,
    NetworkWeights,
    NeuralNetwork,
    _nn_core,
    flatten_weights,
    identity,
    leaky_relu,
    relu,
    sigmoid,
    softmax,
    tanh,
    unflatten_weights,
)
