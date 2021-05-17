import pytest

from f_it import FIt


def exhaust(it):
    for _ in it:
        pass


@pytest.mark.parametrize(["depth"], [[0], [1], [2], [4], [8], [16]])
def test_bench(benchmark, depth):
    # for some reason, iterating over a raw range benchmarks slow
    it = iter(list(range(1000)))
    while depth > 0:
        it = FIt(it)
        depth -= 1

    benchmark(exhaust, it)
