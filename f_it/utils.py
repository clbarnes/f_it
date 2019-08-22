import math


def len_or_none(obj):
    """Return length if available; None otherwise"""
    try:
        return len(obj)
    except TypeError:
        return None


def nCr(n, r, replace=False):
    """Number of combinations, with or without replacement

    :param n: population length
    :param r: sample length
    :param replace: whether to use replacement
    :return: int
    """
    if n < 0 or r < 0:
        raise ValueError("n and r must be positive")
    if n == 0 or r > n:
        return 0

    if replace:
        return math.factorial(n + r - 1) // math.factorial(r) // math.factorial(n - 1)
    else:
        return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


def n_permutations(n, r=None):
    """Number of permutations (unique by position)

    :param n: population length
    :param r: sample length
    :return: int
    """
    if r is None:
        r = n
    if n < 0 or r < 0:
        raise ValueError("n and r must be positive")
    if n == 0 or r > n:
        return 0

    return math.factorial(n) // math.factorial(n - r)
