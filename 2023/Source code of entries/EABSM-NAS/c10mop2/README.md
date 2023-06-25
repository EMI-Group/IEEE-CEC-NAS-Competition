# run the c10mop1 experiment
```
# train the surroagte model and then load the model to perform ea 

# 1.train the offline surroagte model
run the notebook: c10_mop2_predictor.ipynb

# 2.ea
python benchmark_c10_mop2.py
```

# requirements
```
pymoo==0.5.0
```

# minor modifications in pymoo
```
# you can see these modifications 

## in ../pymoo/core/infill.py

# ------------   mod   -------------------
# for task pid=1 with less variety of individuals, generate more individuals.
if (problem.pid == 1) and (n_remaining < 10):
    _off = self._do(problem, pop, n_remaining*10, **kwargs)
else:
    _off = self._do(problem, pop, n_remaining, **kwargs)
# ------------   mod   -------------------

# get the valid and unique individuals
# ------------   mod   -------------------
# filter valid architecture codes and add to searched history
_off = _off[problem._get_fine_x(_off.get("X"))]
problem._add_history(_off.get("X"))
# ------------   mod   -------------------

## in ../pymoo/core/initialization.py

# ------------   mod   -------------------
pop = self.sampling(problem, n_samples, **kwargs)
pop = pop[problem._get_fine_x(pop.get("X"))] # get the valid and unique architectures
problem._add_history(pop.get("X"))
while len(pop) < n_samples:
    pop_t = self.sampling(problem, n_samples - len(pop), **kwargs)
    pop_t = pop_t[problem._get_fine_x(pop_t.get("X"))]      
    problem._add_history(pop_t.get("X"))
    pop = Population.merge(pop, pop_t)
# ------------   mod   -------------------
```


# minor modifications in evoxbenchmark
```
# benchmark_c10_mop2.py:302
hv_pop = benchmark.calc_perf_indicator050(algorithm.pop.get("X"), 'hv')

# we add this function in evoxbench/modules/benchmark.py to calculate the performance indicator in pymoo0.5.0
def calc_perf_indicator050(self, inputs, indicator='igd'):
    import numpy as np
    from numpy import ndarray
    from abc import ABC, abstractmethod

    # The performance indicator that calculates IGD and HV are from [pymoo](https://pymoo.org/).
    try:
        from pymoo.factory import get_performance_indicator
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    except ImportError:
        # Using local pymoo module if import error.
        from evoxbench.utils.pymoo.factory import get_performance_indicator
        from evoxbench.utils.pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    from .evaluator import Evaluator
    from .search_space import SearchSpace

    assert indicator in ['igd', 'hv', 'normalized_hv'], "The requested performance indicator is not supported"

    if indicator == 'igd':
        assert self.pareto_front is not None, "Pareto front needs to be defined before IGD calculation"

    if indicator == 'hv':
        assert self.hv_ref_point is not None, "A reference point need to be defined before HV calculation"

    if indicator == 'igd':
        # normalize Pareto front
        pf = self.normalize(self.pareto_front)
        metric = get_performance_indicator("igd", pf)

    elif 'hv' in indicator:
        hv_ref_point = self.normalize(self.hv_ref_point)
        metric = get_performance_indicator("hv", hv_ref_point)
    else:
        raise KeyError("the requested performance indicator is not define")

    if isinstance(inputs[0], ndarray):
        # re-evaluate the true performance
        F = self.evaluate(inputs, true_eval=True)  # use true/mean accuracy

    else:
        batch_stats = self.evaluator.evaluate(inputs, true_eval=True)
        F = self.to_matrix(batch_stats)

    if not self.normalized_objectives:
        F = self.normalize(F)  # in case the benchmark evaluator does not normalize objs by default

    # filter out the non-dominated solutions
    nd_front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    performance = metric.do(F[nd_front])

    if indicator == 'normalized_hv' and self.pareto_front is not None:
        hv_norm = metric.do(self.normalize(self.pareto_front))
        performance = performance / hv_norm

    return performance
```