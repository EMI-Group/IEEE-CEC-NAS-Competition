import numpy as np
from pymoo.core.problem import Problem

from utils import get_benchmark_settings


class BaseProblem(Problem):
    pid = None
    database = None
    name = None
    n_var = None
    objs = None
    sampling = None
    mutation = None
    crossover = None
    benchmark = None

    def __init__(self):
        xu = np.array(self.benchmark.search_space.ub)
        xl = np.array(self.benchmark.search_space.lb)
        self.pop_size, self.n_gen, self.ref_dirs = get_benchmark_settings(len(self.objs))
        self.off_spring = self.pop_size
        super().__init__(n_var=self.n_var, n_obj=len(self.objs), xl=xl, xu=xu, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X.astype(int)
        F = self.benchmark.evaluate(X=X, true_eval=False)
        out["F"] = F
