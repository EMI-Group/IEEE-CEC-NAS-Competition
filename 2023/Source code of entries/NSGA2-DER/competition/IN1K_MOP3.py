from competition2.base import BaseProblem
from evoxbench.test_suites import in1kmop
from evolution.operations_resnet50 import ResNet50Sampling, ResNet50Mutation, ResNet50Crossover


class IN1K_MOP3(BaseProblem):
    pid = 3
    database = 'ResNet50'
    name = 'IN1K_MOP3'
    n_var = 25
    objs = ['err', 'params', 'flops']
    sampling = ResNet50Sampling()
    mutation = ResNet50Mutation(0.1)
    crossover = ResNet50Crossover()
    benchmark = in1kmop(pid)
