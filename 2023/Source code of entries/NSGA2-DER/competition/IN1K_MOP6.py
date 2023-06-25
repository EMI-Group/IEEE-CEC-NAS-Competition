from competition2.base import BaseProblem
from evoxbench.test_suites import in1kmop
from evolution.operations_transformer import TransformerMutation, TransformerSampling, TransformerCrossover


class IN1K_MOP6(BaseProblem):
    pid = 6
    database = 'Transformer'
    name = 'IN1K_MOP6'
    n_var = 34
    objs = ['err', 'params', 'flops']
    sampling = TransformerSampling()
    mutation = TransformerMutation(0.1)
    crossover = TransformerCrossover()
    benchmark = in1kmop(pid)
