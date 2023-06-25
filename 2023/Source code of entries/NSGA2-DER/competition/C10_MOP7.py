from competition2.base import BaseProblem
from evolution.operations_nb201 import NB201Mutation, NB201Crossover, NB201Sampling
from evoxbench.test_suites import c10mop


class C10_MOP7(BaseProblem):
    pid = 7
    database = 'NASBench201'
    name = 'C10_MOP7'
    n_var = 6
    objs = ['err', 'params', 'flops', 'edgegpu_latency', 'edgegpu_energy', 'eyeriss_latency', 'eyeriss_energy','eyeriss_arithmetic_intensity']
    sampling = NB201Sampling()
    mutation = NB201Mutation(0.1)
    crossover = NB201Crossover()
    benchmark = c10mop(pid)
