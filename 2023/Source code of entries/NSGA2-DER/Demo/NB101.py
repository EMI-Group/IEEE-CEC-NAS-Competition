import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from evoxbench.benchmarks import NASBench101Benchmark
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.algorithm import Algorithm
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.misc import crossover_mask

from utils import folder_create

folder = folder_create(os.path.join('EXP', os.path.basename(__file__)[:-3]))

objs = 'err&params'
benchmark = NASBench101Benchmark(objs=objs, normalized_objectives=False)


class MyRandIntSampling(Sampling):
    def _do(self, problem: Problem, n_samples, **kwargs):
        X = np.random.randint(
            low=problem.xl,
            high=problem.xu + 1,
            size=[n_samples, problem.n_var])
        F = benchmark.evaluate(X=X, true_eval=False)[:, 0]
        bad_num = np.sum(F == 1)
        if bad_num > 0:
            count = 0
            P = []
            while count < bad_num:
                new_X = np.random.randint(
                    low=problem.xl,
                    high=problem.xu + 1,
                    size=[problem.n_var])
                if benchmark.evaluate(X=[new_X], true_eval=False)[0, 0] == 1:
                    continue
                P.append(new_X)
                count += 1
            X[F == 1] = np.array(P)
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


class NasBench101Repair(Repair):

    def __init__(self) -> None:
        super().__init__()

    def _do(self, problem, X, **kwargs):
        X = X.astype(int)
        F = benchmark.evaluate(X=X, true_eval=False)[:, 0]
        bad_num = np.sum(F == 1)
        if bad_num > 0:
            count = 0
            P = []
            while count < bad_num:
                new_X = np.random.randint(
                    low=problem.xl,
                    high=problem.xu + 1,
                    size=[problem.n_var])
                if benchmark.evaluate(X=[new_X], true_eval=False)[0, 0] == 1:
                    continue
                P.append(new_X)
                count += 1
            X[F == 1] = np.array(P)
        return X


class NasBench101(Problem):
    def __init__(self):
        xu = np.ones([26])
        xu[-5:] = 2
        xl = np.zeros_like(xu)
        super().__init__(n_var=26, n_obj=2, xl=xl, xu=xu, vtype=int)
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
    print(np.max(100 - 100 * F[:, 0]))


class NB101DuplicateElimination(DuplicateElimination):

    def __init__(self, epsilon=1e-16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _do(self, pop, other, is_duplicate):
        D = self.calc_dist(pop, other)
        D[np.isnan(D)] = np.inf

        is_duplicate[np.any(D <= self.epsilon, axis=1)] = True
        return is_duplicate


s_pop = 40
gens = 50

problem = NasBench101()
algorithm = NSGA2(
    pop_size=s_pop,
    # n_offsprings=40,
    sampling=MyRandIntSampling(),
    crossover=UniformCrossover(0.3),
    mutation=RandomIntMutation(0.3),
    # eliminate_duplicates=NB101DuplicateElimination(),
    repair=NasBench101Repair())
res = minimize(problem,
               algorithm,
               ('n_gen', gens),
               seed=1,
               callback=show,
               save_history=True,
               verbose=False)
# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
X = res.X if len(res.X.shape) != 1 else res.X[np.newaxis, :]
print(np.max(100 - 100 * benchmark.evaluate(X=X, true_eval=True)[:, 0]))

frames = []
image_list = [os.path.join(folder, f'{i}.png') for i in range(1, gens + 1)]
for image_name in image_list:
    frames.append(imageio.imread(image_name))
imageio.mimsave(os.path.join(folder, f'{s_pop}x{gens}.gif'), frames, 'GIF', duration=0.1)
for image_name in image_list:
    os.remove(image_name)
