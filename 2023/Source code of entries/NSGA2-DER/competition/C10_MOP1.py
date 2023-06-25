from competition2.base import BaseProblem
from evolution.operations_nb101 import NB101Sampling, NB101Mutation, NB101Crossover
from evoxbench.test_suites import c10mop


class C10_MOP1(BaseProblem):
    pid = 1
    database = 'NASBench101'
    name = 'C10_MOP1'
    n_var = 26
    objs = ['err', 'params']
    sampling = NB101Sampling()
    mutation = NB101Mutation(0.1)
    crossover = NB101Crossover()
    benchmark = c10mop(pid)
