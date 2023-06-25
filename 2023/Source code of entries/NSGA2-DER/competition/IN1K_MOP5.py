from competition2.base import BaseProblem
from evoxbench.test_suites import in1kmop
from evolution.operations_transformer import TransformerMutation, TransformerSampling, TransformerCrossover


class IN1K_MOP5(BaseProblem):
    pid = 5
    database = 'Transformer'
    name = 'IN1K_MOP5'
    n_var = 34
    objs = ['err', 'flops']
    sampling = TransformerSampling()
    mutation = TransformerMutation(0.1)
    crossover = TransformerCrossover()
    benchmark = in1kmop(pid)
