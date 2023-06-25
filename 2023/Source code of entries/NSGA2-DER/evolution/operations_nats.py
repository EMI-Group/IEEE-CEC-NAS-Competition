from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.lhs import LatinHypercubeSampling

NATSSampling = LatinHypercubeSampling
NATSMutation = PolynomialMutation
NATSCrossover = SimulatedBinaryCrossover
