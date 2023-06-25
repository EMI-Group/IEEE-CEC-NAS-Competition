from competition2.base import BaseProblem
from evolution.operations_nats import NATSSampling, NATSMutation, NATSCrossover
from evoxbench.test_suites import c10mop


class C10_MOP4(BaseProblem):
    pid = 4
    database = 'NATS'
    name = 'C10_MOP4'
    n_var = 5
    objs = ['err', 'params', 'flops', 'latency']
    sampling = NATSSampling()
    mutation = NATSMutation(0.1)
    crossover = NATSCrossover()
    benchmark = c10mop(pid)
