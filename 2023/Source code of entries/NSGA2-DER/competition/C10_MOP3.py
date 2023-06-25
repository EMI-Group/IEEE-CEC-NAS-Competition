from competition2.base import BaseProblem
from evolution.operations_nats import NATSSampling, NATSMutation, NATSCrossover
from evoxbench.test_suites import c10mop


class C10_MOP3(BaseProblem):
    pid = 3
    database = 'NATS'
    name = 'C10_MOP3'
    n_var = 5
    objs = ['err', 'params', 'flops']
    sampling = NATSSampling()
    mutation = NATSMutation(0.1)
    crossover = NATSCrossover()
    benchmark = c10mop(pid)
