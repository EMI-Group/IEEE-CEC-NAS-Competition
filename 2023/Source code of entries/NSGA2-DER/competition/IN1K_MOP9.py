from competition2.base import BaseProblem
from evoxbench.test_suites import in1kmop
from evolution.operations_mnv3 import MNV3Sampling, MNV3Mutation, MNV3Crossover


class IN1K_MOP9(BaseProblem):
    pid = 9
    database = 'MNV3'
    name = 'IN1K_MOP9'
    n_var = 21
    objs = ['err', 'params', 'flops', 'latency']
    sampling = MNV3Sampling()
    mutation = MNV3Mutation(0.1)
    crossover = MNV3Crossover()
    benchmark = in1kmop(pid)
