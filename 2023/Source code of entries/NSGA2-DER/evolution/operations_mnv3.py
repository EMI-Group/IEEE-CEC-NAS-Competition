# from pymoo.core.mutation import Mutation
# from pymoo.core.sampling import Sampling
# from pymoo.operators.crossover.pntx import TwoPointCrossover
#
# from evolution.operations_base import choice_sampling, choice_mutation
#
#
# class MNV3Sampling(Sampling):
#
#     def __init__(self):
#         super().__init__()
#
#     def _do(self, problem, n_samples, **kwargs):
#         xl, xu = problem.bounds()
#         X = choice_sampling(6, n_samples, xu, xl)
#         return X
#
#
# class MNV3Mutation(Mutation):
#     def __init__(self, prob):
#         super().__init__(prob=prob)
#         self.prob = prob
#
#     def _do(self, problem, X, **kwargs):
#         return choice_mutation(X, up=problem.xu, low=problem.xl)
#
#
# MNV3Crossover = TwoPointCrossover

from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.lhs import LatinHypercubeSampling

MNV3Crossover = SimulatedBinaryCrossover
MNV3Sampling = LatinHypercubeSampling
MNV3Mutation = PolynomialMutation
