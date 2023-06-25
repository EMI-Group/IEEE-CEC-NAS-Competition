from competition2.base import BaseProblem
from evolution.operations_darts import DARTSSampling, DARTSMutation, DARTSCrossover
from evoxbench.test_suites import c10mop


class C10_MOP9(BaseProblem):
    pid = 9
    database = 'DARTS'
    name = 'C10_MOP9'
    n_var = 32
    objs = ['err', 'params', 'flops']
    sampling = DARTSSampling()
    mutation = DARTSMutation(0.1)
    crossover = DARTSCrossover()
    benchmark = c10mop(pid)
