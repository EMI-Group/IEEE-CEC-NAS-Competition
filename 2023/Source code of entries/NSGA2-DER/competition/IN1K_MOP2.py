from competition2.base import BaseProblem
from evoxbench.test_suites import in1kmop
from evolution.operations_resnet50 import ResNet50Sampling, ResNet50Mutation, ResNet50Crossover


class IN1K_MOP2(BaseProblem):
    pid = 2
    database = 'ResNet50'
    name = 'IN1K_MOP2'
    n_var = 25
    objs = ['err', 'flops']
    sampling = ResNet50Sampling()
    mutation = ResNet50Mutation(0.1)
    crossover = ResNet50Crossover()
    benchmark = in1kmop(pid)
