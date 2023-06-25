from copy import deepcopy

import numpy as np


def binary_sampling(n_var, n_samples, rate_for_1=0.5):
    """
    Generate binary population.
    :param n_var: Number of dimension of each individual
    :param n_samples: Number of samples
    :param rate_for_1: Rate of '1' in each individual
    :return: Individuals (np.ndarray) of shape [n_samples,n_var]
    """
    return (np.random.random([n_samples, n_var]) < rate_for_1).astype(int)


def choice_sampling(n_var, n_samples, up, low):
    """
    Generate population where each element are chose from given numbers.
    :param n_var: Number of dimension of each individual
    :param n_samples: Number of samples
    :param up: Upper bound (included), it can be a integer or a ndarray with shape [n_var]
    :param low: Lower bound (included), it can be a integer or a ndarray with shape [n_var]
    :return: Individuals (np.ndarray) of shape [n_samples,n_var]
    """
    x = np.random.random([n_samples, n_var]) * (up + 1 - low) + low
    return x.astype(int)


def values_sampling(n_var, n_samples, up, low, need_int=False):
    """
    Generate population use latin-hypercube.
    If you need integer, you should control the bounds out of this function.
    :param n_var: Number of dimension of each individual
    :param n_samples: Number of samples
    :param up: Upper bound (excluded), it can be a integer or a ndarray with shape [n_var]
    :param low: Lower bound (excluded), it can be a integer or a ndarray with shape [n_var]
    :param need_int: Type of dimension is integer or not. If true, floor() is applied for every element.
    :return: Individuals (np.ndarray) of shape [n_samples,n_var]
    """

    # Initialize an empty array to store the samples
    samples = np.zeros((n_samples, n_var))
    # Divide the range [0, 1) into n_samples equally sized intervals
    intervals = np.linspace(0, 1, n_samples + 1)
    # Generate a random permutation of the integers 0 through n_samples-1
    perm = np.random.permutation(n_samples)
    # Generate n_samples random numbers for each variable in [0, 1)
    rand_nums = np.random.rand(n_samples, n_var)

    # Fill in each column of the sample array with the random numbers
    # in a way that ensures each row has exactly one value in each interval
    for i in range(n_var):
        for j in range(n_samples):
            samples[j, i] = intervals[perm[j]] + rand_nums[j, i] * (intervals[perm[j] + 1] - intervals[perm[j]])

    # Rescale the sample array to the desired range [a, b]
    samples = low + (up - low) * samples

    return samples.astype(int) if need_int else samples


def binary_mutation(x, rate=0.5):
    _x = deepcopy(x)
    mask = np.random.random(_x.shape) < rate
    _x[mask] = 1 - _x[mask]
    return _x


# def binary_crossover(x1, x2, rate=0.5):
#     assert x1.shape == x2.shape
#     _x1 = deepcopy(x1)
#     _x2 = deepcopy(x2)
#     mask = np.random.random(_x1.shape) < rate
#     _x1[mask], _x2[mask] = _x2[mask], _x1[mask]
#     return _x1, _x2


def choice_mutation(x, up, low, rate=0.5):
    _x = deepcopy(x)
    up = np.ones(x.shape) * up
    low = np.ones(x.shape) * low
    mask = np.random.random(_x.shape) < rate
    diff = np.random.randint(up - low, size=mask.shape)
    _x[mask] += diff[mask]
    _x[_x > up] = _x[_x > up] - (up - low)[_x > up]
    return _x


# def choice_crossover(x1, x2, rate=0.5):
#     return binary_crossover(x1, x2, rate)

if __name__ == '__main__':
    # print(
    #     binary_sampling(n_samples=3, n_var=20, rate_for_1=0.6)
    # )
    # print(
    #     choice_sampling(n_samples=3, n_var=20, up=10, low=0)
    # )
    # print(
    #     choice_sampling(n_samples=5, n_var=20, up=np.array([10] * 10 + [20] * 10), low=0)
    # )
    # print(
    #     values_sampling(n_samples=3, n_var=20, up=10, low=0, need_int=True)
    # )
    # print(
    #     values_sampling(n_samples=5, n_var=20, up=np.array([10] * 10 + [20] * 10), low=0)
    # )
    # d1 = binary_sampling(n_samples=1, n_var=20, rate_for_1=0.6)
    # d2 = binary_sampling(n_samples=1, n_var=20, rate_for_1=0.6)
    # a1, a2 = binary_crossover(d1, d2)
    # print(d1, d2)
    # print(a1, a2)
    up = np.array([10] * 10 + [20] * 10)
    d1 = choice_sampling(n_samples=5, n_var=20, up=up, low=0)
    a1 = choice_mutation(d1, up=up, low=0)
    print(d1, a1)
