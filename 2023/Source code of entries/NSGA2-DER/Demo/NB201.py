import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from evoxbench.benchmarks import NASBench201Benchmark
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.algorithm import Algorithm
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.misc import crossover_mask
from pymoo.util.ref_dirs import get_reference_directions

from utils import folder_create

folder = folder_create(os.path.join('EXP', os.path.basename(__file__)[:-3]))

objs = 'err&params&flops'
benchmark = NASBench201Benchmark(objs=objs, normalized_objectives=False)


class MyRandIntSampling(Sampling):
    def _do(self, problem: Problem, n_samples, **kwargs):
        X = np.random.randint(
            low=problem.xl,
            high=problem.xu + 1,
            size=[n_samples, problem.n_var])
        return X


class UniformCrossover(Crossover):
    def __init__(self, prob=0.5, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.prob = prob

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < self.prob
        _X = crossover_mask(X, M)
        return _X


class RandomIntMutation(Mutation):
    def __init__(self, prob=0.05, prob_var=None, **kwargs) -> None:
        super().__init__(prob, prob_var, **kwargs)
        self.prob = prob

    def _do(self, problem: Problem, X, **kwargs):
        mask = np.random.rand(*X.shape)
        r = np.random.randint(
            low=problem.xl,
            high=problem.xu + 1,
            size=X.shape)
        X[mask < self.prob] = r[mask < self.prob]
        return X


class NasBench201(Problem):
    database = 'NASBench101'
    name = 'C10_MOP2'
    n_var = 26
    objs = ['err', 'params', 'flops']
    pop_size = 105
    off_spring = 100
    n_gen = 100
    def __init__(self):
        xu = np.ones([6]) * 4
        xl = np.zeros_like(xu)
        super().__init__(n_var=6, n_obj=2, xl=xl, xu=xu, vtype=int)
        # self.elementwise = True

    def _evaluate(self, x, out, *args, **kwargs):
        F = benchmark.evaluate(X=x, true_eval=False)
        assert np.all(F[:, 0] != 1)
        out["F"] = F


def show(algorithm: Algorithm):
    with open(os.path.join(folder, 'history.txt'), 'a+') as f:
        for i in algorithm.pop:
            f.write(f'{100 - i.F[0]:5.2f},')
        f.write('\n')
    X = [i.X for i in algorithm.pop]
    F = benchmark.evaluate(X=X, true_eval=True)
    plt.scatter(F[:, 0], F[:, 1])
    plt.title(f'{algorithm.n_gen}')
    plt.savefig(os.path.join(folder, f'{algorithm.n_gen}.png'))
    plt.cla()
    print(np.max(100 - F[:, 0]))


s_pop = 20
n_gen = 40

problem = NasBench201()
ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=13)
algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=problem.pop_size, sampling=MyRandIntSampling(),
    crossover=UniformCrossover(0.3),
    mutation=RandomIntMutation(0.3), eliminate_duplicates=True)
res = minimize(problem, algorithm, ('n_gen', n_gen), verbose=True)

# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
X = res.X if len(res.X.shape) != 1 else res.X[np.newaxis, :]
print(np.max(100 - benchmark.evaluate(X=X, true_eval=True)[:, 0]))

frames = []
image_list = [os.path.join(folder, f'{i}.png') for i in range(1, gens + 1)]
for image_name in image_list:
    frames.append(imageio.imread(image_name))
imageio.mimsave(os.path.join(folder, f'{s_pop}x{gens}.gif'), frames, 'GIF', duration=0.1)
for image_name in image_list:
    os.remove(image_name)
