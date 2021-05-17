from functools import reduce
from itertools import (
    chain,
    combinations,
    combinations_with_replacement,
    compress,
    count,
    cycle,
    dropwhile,
    filterfalse,
    groupby,
    islice,
    permutations,
    product,
    starmap,
    takewhile,
    zip_longest,
)
from unittest import mock

import pytest

from f_it import FIt
from f_it.fit import ensure_FIt
from f_it.utils import len_or_none


def assert_equal_it(it1, it2):
    len1 = len_or_none(it1)
    len2 = len_or_none(it2)

    if len1 is not None and len2 is not None:
        assert len1 == len2

    count = 0

    for item1, item2 in zip_longest(it1, it2):
        assert item1 == item2
        count += 1

    if len1 is not None:
        assert count == len1
    if len2 is not None:
        assert count == len2


def test_ensure_FIt(lst):
    assert isinstance(ensure_FIt(lst), FIt)
    fiter = FIt(lst)
    assert ensure_FIt(fiter) is fiter


def test_instantiate(lst):
    it = FIt(lst)
    assert it.length == len(it) == len(lst)


def test_instantiate_nolen(lst):
    it = FIt(iter(lst))
    assert it.length is None
    with pytest.raises(TypeError):
        len(it)


def test_instantiate_explicit_len(lst):
    it = FIt(iter(lst), len(lst))
    assert it.length == len(it) == len(lst)


def test_next():
    fiter = FIt([1])
    fiter2 = FIt([1])

    assert fiter.next() == next(fiter2)
    with pytest.raises(StopIteration):
        fiter.next()


def test_to(lst):
    assert FIt(lst).to(set) == set(lst)


def test_add(fiter, lst):
    new = fiter + iter(lst)
    assert list(new) == lst * 2


def test_radd(fiter, lst):
    new = iter(lst) + fiter
    assert list(new) == lst * 2


def test_getitem(lst):
    fiter = FIt(lst)
    assert fiter[2] == lst[2]


def test_getitem_neg(lst):
    fiter = FIt(lst)
    assert fiter[-1] == lst[-1]


def test_getitem_neg_nolen(lst):
    fiter = FIt(iter(lst))
    with pytest.raises(ValueError):
        fiter[-1]


def test_getitem_slice(fiter, lst):
    assert list(fiter[2:12:3]) == lst[2:12:3]


# builtin


def test_enumerate(fiter, lst):
    assert_equal_it(fiter.enumerate(), enumerate(lst))


def test_filter(fiter, lst):
    def fn(x):
        return x % 2

    assert_equal_it(fiter.filter(fn), filter(fn, lst))


def test_map(fiter, lst):
    def fn(x):
        return x ** 3

    assert_equal_it(fiter.map(fn), map(fn, lst))


@pytest.mark.parametrize(["args"], [((10,),), ((2, 20, 5),)])
def test_range(args):
    assert_equal_it(FIt.range(*args), range(*args))


def test_zip(fiter, lst):
    zip_with = list(range(10, 10 + len(lst)))
    assert_equal_it(fiter.zip(zip_with), zip(lst, zip_with))


def test_zip_short(fiter, lst):
    zip_with = list(range(10, 10 + len(lst)))
    zip_with = zip_with[:-2]
    assert_equal_it(fiter.zip(zip_with), zip(lst, zip_with))


def test_zip_long(fiter, lst):
    zip_with = list(range(10, 10 + len(lst)))
    zip_with = zip_with * 2
    assert_equal_it(fiter.zip(zip_with), zip(lst, zip_with))


def test_zip_long_longest(fiter, lst):
    zip_with = list(range(len(lst) * 2))
    assert_equal_it(fiter.zip(zip_with, longest=True), zip_longest(lst, zip_with))


# functools


def test_reduce(fiter, lst):
    def fn(x, y):
        return x + y

    assert fiter.reduce(fn) == reduce(fn, lst)


def test_reduce_initialiser(fiter, lst):
    def fn(x, y):
        return x + y

    assert fiter.reduce(fn, 10) == reduce(fn, lst, 10)


# itertools


def test_chain(fiter, lst):
    append = [1, 2, 3]
    assert_equal_it(fiter.chain(append), chain(lst, append))


def test_chain_from_iterable(fiter, lst):
    two = (lst.copy() for _ in range(2))
    three = (lst.copy() for _ in range(3))
    assert_equal_it(fiter.chain_from_iterable(two), chain.from_iterable(three))


def test_combinations(fiter, lst):
    init_len = len_or_none(fiter)
    combs = fiter.combinations(3)
    if init_len is not None:
        assert len_or_none(combs) is not None
    assert_equal_it(combs, combinations(lst, 3))


def test_combinations_replace(fiter, lst):
    init_len = len_or_none(fiter)
    combs = fiter.combinations(3, True)

    if init_len is not None:
        assert len_or_none(combs) is not None

    assert_equal_it(combs, combinations_with_replacement(lst, 3))


def test_combinations_with_replacement(fiter, lst):
    init_len = len_or_none(fiter)
    combs = fiter.combinations_with_replacement(3)

    if init_len is not None:
        assert len_or_none(combs) is not None

    assert_equal_it(combs, combinations_with_replacement(lst, 3))


def test_compress(fiter, lst):
    compressor = [bool(item % 2) for item in lst]
    assert_equal_it(fiter.compress(compressor), compress(lst, compressor))


def test_count(fiter, lst):
    fiter = FIt.count(2, 20)
    other = count(2, 20)
    assert_equal_it(islice(fiter, 20), islice(other, 20))


def test_cycle(fiter, lst):
    cycling = fiter.cycle(3)
    if len_or_none(fiter):
        assert len(cycling) == len(lst) * 3
    assert_equal_it(cycling, lst * 3)


def test_cycle_none(fiter, lst):
    assert_equal_it(islice(fiter.cycle(), 1000), islice(cycle(lst), 1000))


def test_dropwhile(fiter, lst):
    def pred(x):
        return x < 3

    assert_equal_it(fiter.dropwhile(pred), dropwhile(pred, lst))


def test_filterfalse(fiter, lst):
    def pred(x):
        return bool(x % 3)

    assert_equal_it(fiter.filterfalse(pred), filterfalse(pred, lst))


def test_groupby(fiter, lst):
    def key(x):
        return x // 5

    for (k1, v1), (k2, v2) in zip(fiter.groupby(key), groupby(lst, key)):
        assert k1 == k2
        assert_equal_it(v1, v2)


def test_islice(fiter, lst):
    assert_equal_it(fiter.islice(5, 10), lst[5:10])


def test_islice_neg(lst):
    fiter = FIt(lst)
    assert_equal_it(fiter.islice(5, -2), lst[5:-2])


def test_permutations(fiter, lst):
    assert_equal_it(fiter.permutations(3), permutations(lst, 3))


def test_product(fiter, lst):
    prod_with = ([1, 2], [3, 4])
    r = 2
    assert_equal_it(
        fiter.product(*prod_with, repeat=r), product(lst, *prod_with, repeat=r)
    )


def test_repeat():
    it = FIt.repeat(1, 5)
    assert len(it) == 5
    assert list(it) == [1] * 5


def test_starmap():
    lst = [(x, x ** 2) for x in range(10)]
    fiter = FIt(lst)

    def fn(x, y):
        return x + y

    assert_equal_it(fiter.starmap(fn), starmap(fn, lst))


def test_takewhile(fiter, lst):
    def pred(x):
        return x < 5

    assert_equal_it(fiter.takewhile(pred), takewhile(pred, lst))


def test_takewhile_nopred(fiter, lst):
    assert_equal_it(fiter.takewhile(), takewhile(bool, lst))


def test_tee(fiter):
    fit1, fit2 = fiter.tee()
    assert_equal_it(fit1, list(fit2))


def test_zip_longest(fiter, lst):
    extended = lst + [1, 2, 3, 4]
    assert_equal_it(fiter.zip_longest(extended), zip_longest(lst, extended))


def test_zip_longest_fill(fiter, lst):
    extended = lst + [1, 2, 3, 4]
    assert_equal_it(
        fiter.zip_longest(extended, fill_value=5),
        zip_longest(lst, extended, fillvalue=5),
    )


# recipes


def test_take(fiter, lst):
    assert fiter.take(5) == lst[:5]


def test_tail(fiter, lst):
    assert_equal_it(fiter.tail(10), lst[-10:])


def test_consume(fiter, lst):
    assert_equal_it(fiter.consume(5), lst[5:])


def test_consume_all(fiter, lst):
    assert_equal_it(fiter.consume(), [])


def test_nth(fiter, lst):
    assert fiter.nth(5) == lst[5]


def test_flatten():
    fiter = FIt([1, 2, [3, 4], [5, [6, 7]]])
    assert_equal_it(fiter.flatten(), [1, 2, 3, 4, 5, 6, 7])


def test_flatten_levels():
    fiter = FIt([1, 2, [3, 4], [5, [6, 7]]])
    assert_equal_it(fiter.flatten(levels=1), [1, 2, 3, 4, 5, [6, 7]])


def test_flatten_str():
    fiter = FIt([1, [2, "potato"]])
    assert_equal_it(fiter.flatten(), [1, 2, "potato"])


def test_flatten_nostr():
    fiter = FIt([1, [2, "potato"]])
    assert_equal_it(fiter.flatten(split_strings=True), [1, 2] + list("potato"))


def test_flatten_0():
    lst = [1, 2, [3, 4], [5, [6, 7]]]
    fiter = FIt(lst)
    assert_equal_it(fiter.flatten(0), lst)


# others


def test_sliding_window():
    fiter = FIt(range(5))
    expected = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert_equal_it(fiter.sliding_window(3), expected)

    fiter2 = FIt(range(5))
    assert_equal_it(fiter2.sliding_window(10), [])


def test_chunk():
    fiter = FIt(range(5))
    expected = [[0, 1], [2, 3], [4]]
    assert_equal_it(fiter.chunk(2), expected)


@pytest.mark.parametrize(["wrapped"], [(range(5),), (iter(range(5)),)])
def test_interleave(wrapped):
    fiter = FIt(wrapped)
    mix_with = range(5, 10)
    expected = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    assert_equal_it(fiter.interleave(mix_with), expected)


def test_peek(fiter, lst):
    initial_len = len_or_none(fiter)
    assert fiter.peek(5) == lst[:5]
    assert initial_len == len_or_none(fiter)
    assert fiter.peek() == lst[0]
    assert_equal_it(fiter, lst)
    assert fiter.peek(0) == []


def test_progress_notqdm(fiter, lst, monkeypatch):
    monkeypatch.delattr("tqdm.tqdm")
    with pytest.warns(UserWarning):
        assert_equal_it(fiter.progress(), lst)


def test_progress(fiter, lst):
    with mock.patch("tqdm.tqdm") as MockTqdm:
        MockTqdm.side_effect = lambda x: (i for i in x)
        assert_equal_it(fiter.progress(), lst)
        MockTqdm.assert_called_once()


def test_for_each(fiter, lst):
    outer = []

    def fn(item):
        outer.append(item)

    fiter.for_each(fn)

    assert outer == lst


def test_length():
    fiter = FIt(range(5))
    assert len(fiter) == 5
    next(fiter)
    assert len(fiter) == 4
    fiter.peek()
    assert len(fiter) == 4


@pytest.mark.parametrize(
    ["lst", "expected"],
    [
        ([0, 0, 0], False),
        ([0, 0, 1], True),
        ([], False),
    ],
)
def test_any(lst, expected):
    fiter = FIt(lst)
    assert fiter.any() == expected


@pytest.mark.parametrize(
    ["lst", "expected"],
    [
        ([1, 1, 1], True),
        ([1, 1, 0], False),
        ([], True),
    ],
)
def test_all(lst, expected):
    fiter = FIt(lst)
    assert fiter.all() == expected
