from competition2.base import BaseProblem
from evolution.operations_nb101 import NB101Sampling, NB101Mutation, NB101Crossover
from evoxbench.test_suites import c10mop


class C10_MOP2(BaseProblem):
    pid = 2
    database = 'NASBench101'
    name = 'C10_MOP2'
    n_var = 26
    objs = ['err', 'params', 'flops']
    sampling = NB101Sampling()
    mutation = NB101Mutation(0.1)
    crossover = NB101Crossover()
    benchmark = c10mop(pid)
