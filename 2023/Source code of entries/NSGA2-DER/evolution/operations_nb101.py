import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.pntx import TwoPointCrossover


from evolution.operations_base import binary_sampling, choice_sampling, choice_mutation


class NB101Sampling(Sampling):

    def __init__(self):
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        X = np.concatenate([binary_sampling(21, n_samples), choice_sampling(5, n_samples, 2, 0)], axis=1)
        return X


class NB101Mutation(Mutation):
    def __init__(self, prob):
        super().__init__(prob=prob)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        Xp = np.copy(X)
        flip = np.random.random([X.shape[0],21]) < self.prob
        Xp[:,:21][flip] = 1 - X[:,:21][flip]
        Xp[:,21:]=choice_mutation(Xp[:,21:], up=2, low=0)
        return Xp


NB101Crossover=TwoPointCrossover
