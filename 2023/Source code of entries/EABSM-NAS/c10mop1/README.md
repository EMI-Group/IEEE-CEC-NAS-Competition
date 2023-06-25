# run the c10mop1 experiment
```
# ea
python benchmark_c10_mop1.py

# train the surroagte model and perform local search
run the notebook: c10_mop1_predictor.ipynb
```

# requirements
```
pymoo==0.6.0
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
