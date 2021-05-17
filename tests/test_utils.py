import pytest

from f_it.utils import len_or_none, n_permutations, nCr


def test_len_or_none(lst):
    length = len(lst)
    assert len_or_none(lst) == length
    assert len_or_none(iter(lst)) is None


def test_nCr():
    assert nCr(3, 2) == 3


def test_nCr_edges():
    with pytest.raises(ValueError):
        nCr(-1, 1)
    with pytest.raises(ValueError):
        nCr(1, -1)
    assert nCr(0, 0) == 0
    assert nCr(2, 3) == 0


def test_nCr_replace():
    assert nCr(3, 2, True) == 6


def test_n_permutations():
    assert n_permutations(3, 2) == 6


def test_n_permutations_edges():
    with pytest.raises(ValueError):
        n_permutations(-1, 1)
    with pytest.raises(ValueError):
        n_permutations(1, -1)
    assert n_permutations(0, 0) == 0
    assert n_permutations(2, 3) == 0
    assert n_permutations(5) == n_permutations(5, 5)
