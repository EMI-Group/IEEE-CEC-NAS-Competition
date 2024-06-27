import numpy as np
from numpy import ndarray

def compare_f1_f2(f1: ndarray, f2: ndarray) -> int:
    """
    Takes in the objective function values of two solution (f1, f2). Returns the better one using Pareto-dominance definition.

    :param f1: the objective function values of the first solution
    :param f2: the objective function values of the second solution
    :return: -1 (no one is better); 0 (f1 is better); or 1 (f2 is better)
    """
    x_better = np.all(f1 <= f2)
    y_better = np.all(f2 <= f1)
    if x_better == y_better:
        return -1
    if y_better:  # False - True
        return 1
    return 0  # True - False


def is_equal(f1: ndarray, f2: ndarray) -> bool:
    """
    Takes in the objective function values of two solution (f1, f2.)
    Returns the better one using Pareto-dominance definition.

    :param f1: the objective function values of the first solution
    :param f2: the objective function values of the second solution
    :return: True or False
    """
    return np.all(f1 == f2)

def not_existed(genotypeHash: str, **kwargs) -> bool:
    """
    Takes in the fingerprint of a solution and a set of checklists.
    Return True if the current solution have not existed on the set of checklists.

    :param genotypeHash: the fingerprint of the considering solution
    :return: True or False
    """
    return np.all([genotypeHash not in kwargs[L] for L in kwargs])


