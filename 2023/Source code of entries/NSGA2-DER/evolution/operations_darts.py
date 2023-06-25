from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.pntx import TwoPointCrossover

from evolution.operations_base import choice_sampling, choice_mutation


class DARTSSampling(Sampling):

    def __init__(self):
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        X = choice_sampling(32, n_samples, xu, xl)
        return X


class DARTSMutation(Mutation):
    def __init__(self, prob):
        super().__init__(prob=prob)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        return choice_mutation(X, up=problem.xu, low=problem.xl)


DARTSCrossover = TwoPointCrossover
