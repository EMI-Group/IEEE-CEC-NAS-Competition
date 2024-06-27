import numpy as np

# RESULT
class Result:
    def __init__(self, genotype_list):
        self.genotype_list = np.array(genotype_list)

class Footprint:
    def __init__(self):
        self.data = {}

class Debugger:
    def __init__(self, **kwargs):
        self.verbose = kwargs['verbose']

    def __call__(self, **kwargs):
        algorithm = kwargs['algorithm']
        print('#Evals:', algorithm.evaluator.n_eval)
