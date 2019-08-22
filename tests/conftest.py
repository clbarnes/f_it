import pytest

from f_it import FIt
from f_it.utils import len_or_none

LEN = 20


@pytest.fixture
def lst():
    return list(range(LEN))


@pytest.fixture(params=["list", "nolen"])
def fiter(request, lst):
    if request.param == "list":
        fiter = FIt(lst)
        assert len_or_none(fiter) is not None
        return fiter
    else:
        fiter = FIt(iter(lst))
        assert len_or_none(fiter) is None
        return fiter
