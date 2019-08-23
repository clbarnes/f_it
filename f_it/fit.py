# -*- coding: utf-8 -*-
import math
import warnings
from collections.abc import Iterator
from collections import deque
from itertools import (
    zip_longest,
    islice,
    starmap,
    takewhile,
    tee,
    chain,
    compress,
    dropwhile,
    filterfalse,
    groupby,
    repeat,
    product,
    combinations,
    combinations_with_replacement,
    permutations,
    cycle,
    count,
)
from typing import Iterable, Optional, Callable, Any, List

from f_it.utils import len_or_none, nCr, n_permutations


def ensure_FIt(obj):
    if isinstance(obj, FIt):
        return obj
    else:
        return FIt(obj)


class FIt(Iterator):
    def __init__(self, iterable: Iterable, length=None):
        """Iterator class providing many postfix functional methods.

        Most of these methods can also be used as static methods which take any
        iterable as the first argument.
        Where possible, the returned FIt instances have a length, which expresses how
        many items *remain* in the iterator.

        Where possible, all iteration is evaluated lazily.

        :param iterable: iterable to wrap
        :param length: explicitly provide a length if you know it but the iterable doesn't
        """  # noqa
        self.iterator = iter(iterable)
        if length is None:
            self.init_length = len_or_none(iterable)
        else:
            self.init_length = int(length)

        self.consumed = 0
        self.cache = deque()

    @property
    def length(self) -> Optional[int]:
        """If available, return the number of items remaining.

        Returns None otherwise.
        """
        if self.init_length is None:
            return None
        else:
            return self.init_length - self.consumed + len(self.cache)

    def __len__(self) -> int:
        length = self.length
        if length is None:
            raise TypeError(f"object of type '{type(self).__name__}' has no len()")
        else:
            return length

    def __next__(self):
        if self.cache:
            return self.cache.popleft()
        else:
            item = next(self.iterator)
            self.consumed += 1
            return item

    def next(self):
        """Return the next item in the iterator"""
        return next(self)

    def to(self, fn: Callable[..., Any], *args, **kwargs):
        """Apply a callable to the remainder of the iterator.

        e.g. convert the FIt into a ``list``, find the ``sum`` etc.

        :param fn: callable which takes 1 iterable argument, plus others to be passed with *args, **kwargs
        :param args: passed to the callable after the iterable
        :param kwargs: passed to the callable
        :return:
        """  # noqa
        return fn(self, *args, **kwargs)

    ##################
    # Free functions #
    ##################

    def enumerate(self, start=0):
        """See the builtin enumerate_

        .. _enumerate: https://docs.python.org/3/library/functions.html#enumerate
        """
        return FIt(enumerate(self, start), len_or_none(self))

    def filter(self, function=None):
        """See the builtin filter_

        .. _filter: https://docs.python.org/3/library/functions.html#filter
        """
        return FIt(filter(function, self))

    def map(self, function, *iterables, longest=True):
        """See the builtin map_

        Difference from stdlib: accepts ``longest`` kwarg, for whether to terminate
        with the  longest iterable (padding the others with ``None``) instead of
        terminating with the shortest.

        .. _map: https://docs.python.org/3/library/functions.html#map
        """
        return FIt.zip(self, *iterables, longest=longest).starmap(function)

    @classmethod
    def range(cls, start, stop=None, step=1):
        """See the builtin range_

        .. _range: https://docs.python.org/3/library/functions.html#func-range
        """
        if stop is None:
            stop = start
            start = 0

        return FIt(range(start, stop, step))

    def zip(self, *iterables, longest=False, fill_value=None):
        """See the builtin zip_

        Difference from stdlib: accepts ``longest`` kwarg, which makes this method
        act like itertools.zip_longest_ (also accepts ``fill_value``)

        .. _zip: https://docs.python.org/3/library/functions.html#zip
        .. _zip_longest: https://docs.python.org/3/library/itertools.html#itertools.zip_longest
        """  # noqa

        iterables = [self] + list(iterables)
        if longest:
            return FIt.zip_longest(*iterables, fill_value=fill_value)

        try:
            length = min(len_or_none(it) for it in iterables)
        except TypeError:
            length = None

        return FIt(zip(*iterables), length)

    #############
    # functools #
    #############

    def reduce(self, function, initializer=None):
        """See functools.reduce_

        Difference from stdlib: if ``initializer`` is ``None``, method will behave as if
        it was not given.

        .. _functools.reduce: https://docs.python.org/3/library/functools.html#functools.reduce
        """  # noqa
        it = iter(self)
        if initializer is None:
            value = next(it)
        else:
            value = initializer
        for element in it:
            value = function(value, element)
        return value

    #############
    # itertools #
    #############

    def chain(self, *iterables):
        """See itertools.chain_

        .. _itertools.chain: https://docs.python.org/3/library/itertools.html#itertools.chain
        """  # noqa
        iterables = [self] + list(iterables)
        lengths = [len_or_none(it) for it in iterables]
        try:
            length = sum(lengths)
        except TypeError:
            length = None

        return FIt(chain.from_iterable(iterables), length)

    def chain_from_iterable(self, iterable):
        """See itertools.chain.from_iterable_

        .. _itertools.chain.from_iterable: https://docs.python.org/3/library/itertools.html#itertools.chain.from_iterable
        """  # noqa

        def gen():
            yield from self
            for inner in iterable:
                yield from inner

        return FIt(gen())

    def combinations(self, r: int, replace=False):
        """See itertools.combinations_

        Difference from stdlib: ``replace`` argument to use
        itertools.combinations_with_replacement_

        .. _itertools.combinations: https://docs.python.org/3/library/itertools.html#itertools.combinations
        .. _itertools.combinations_with_replacement: https://docs.python.org/3/library/itertools.html#itertools.combinations_with_replacement
        """  # noqa
        if replace:
            return FIt.combinations_with_replacement(self, r)

        this_len = len_or_none(self)
        if this_len is None:
            length = None
        else:
            length = nCr(this_len, r)

        return FIt(combinations(self, r), length)

    def combinations_with_replacement(self, r: int):
        """See itertools.combinations_with_replacement_

        .. _itertools.combinations_with_replacement: https://docs.python.org/3/library/itertools.html#itertools.combinations_with_replacement
        """  # noqa
        this_len = len_or_none(self)
        if this_len is None:
            length = None
        else:
            length = nCr(this_len, r, True)

        return FIt(combinations_with_replacement(self, r), length)

    def compress(self, selectors: Iterable):
        """See itertools.compress_

        .. _itertools.compress: https://docs.python.org/3/library/itertools.html#itertools.compress
        """  # noqa
        return FIt(compress(self, selectors))

    @classmethod
    def count(cls, start: int = 0, step: int = 1):
        """See itertools.count_

        .. _itertools.count: https://docs.python.org/3/library/itertools.html#itertools.count
        """  # noqa
        return FIt(count(start, step))

    def cycle(self, n: Optional[int] = None):
        """See itertools.cycle_

        Difference from stdlib: accepts ``n`` argument, for how many times it should be
        repeated.

        .. _itertools.cycle: https://docs.python.org/3/library/itertools.html#itertools.cycle
        """  # noqa
        this_len = len_or_none(self)

        if n is None:
            n = float("inf")
        elif this_len is not None:
            length = n * this_len
            return FIt(cycle(self), length).islice(length)

        def gen(it):
            cached = []
            for item in it:
                cached.append(item)
                yield item
            count = 1
            while count < n:
                yield from cached
                count += 1

        return FIt(gen(self))

    def dropwhile(self, predicate: Optional[Callable[[Any], bool]] = None):
        """See itertools.dropwhile_

        Difference from stdlib: if predicate is None (default), use ``bool``

        .. _itertools.dropwhile: https://docs.python.org/3/library/itertools.html#itertools.dropwhile
        """  # noqa
        predicate = predicate or bool
        return FIt(dropwhile(predicate, self))

    def filterfalse(self, predicate: Callable[[Any], bool] = None):
        """See itertools.filterfalse_

        Difference from stdlib: if predicate is None (default), use ``bool``

        .. _itertools.filterfalse: https://docs.python.org/3/library/itertools.html#itertools.filterfalse
        """  # noqa
        predicate = predicate or bool
        return FIt(filterfalse(predicate, self))

    def groupby(self, key: Optional[Callable[[Any], Any]] = None):
        """See itertools.groupby_

        .. _itertools.groupby: https://docs.python.org/3/library/itertools.html#itertools.groupby
        """  # noqa
        return FIt(groupby(self, key))

    def islice(self, start: int, stop: Optional[int] = None, step: int = 1):
        """See itertools.islice_

        .. _itertools.islice: https://docs.python.org/3/library/itertools.html#itertools.islice
        """  # noqa
        if stop is None:
            stop = start
            start = 0

        this_len = len_or_none(self)
        if this_len is None:
            length = None
        else:
            length = max(math.ceil((min(stop, this_len) - start) / step), 0)

        return FIt(islice(self, start, stop, step), length)

    def permutations(self, r: Optional[int] = None):
        """See itertools.permutations_

        .. _itertools.permutations: https://docs.python.org/3/library/itertools.html#itertools.permutations
        """  # noqa
        this_len = len_or_none(self)
        if this_len is None:
            length = None
        else:
            length = n_permutations(this_len, r)

        return FIt(permutations(self, r), length)

    def product(self, *iterables, repeat: int = 1):
        """See itertools.product_

        .. _itertools.product: https://docs.python.org/3/library/itertools.html#itertools.product
        """  # noqa
        iterables = [self] + list(iterables)
        try:
            length = (
                FIt(iterables).map(len_or_none).reduce(lambda x, y: x * y) ** repeat
            )
        except TypeError:
            length = None

        return FIt(product(*iterables, repeat=repeat), length)

    @classmethod
    def repeat(cls, obj, n: Optional[int] = None):
        """See itertools.repeat_

        .. _itertools.repeat: https://docs.python.org/3/library/itertools.html#itertools.repeat
        """  # noqa
        it = repeat(obj) if n is None else repeat(obj, n)
        return cls(it, n)

    def starmap(self, function: Callable):
        """See itertools.starmap_

        .. _itertools.starmap: https://docs.python.org/3/library/itertools.html#itertools.starmap
        """  # noqa
        return FIt(starmap(function, self), len_or_none(self))

    def takewhile(self, predicate: Optional[Callable[[Any], bool]] = None):
        """See itertools.takewhile_

        .. _itertools.takewhile: https://docs.python.org/3/library/itertools.html#itertools.takewhile
        """  # noqa
        if predicate is None:
            predicate = bool
        return FIt(takewhile(predicate, self))

    def tee(self, n: int = 2):
        """See itertools.tee_

        .. _itertools.tee: https://docs.python.org/3/library/itertools.html#itertools.tee
        """  # noqa
        length = len_or_none(self)
        return [FIt(t, length) for t in tee(self, n)]

    def zip_longest(self, *iterables, fill_value=None):
        """See itertools.zip_longest_

        .. _itertools.zip_longest: https://docs.python.org/3/library/itertools.html#itertools.zip_longest
        """  # noqa
        iterables = [self] + list(iterables)

        try:
            length = max(len_or_none(it) for it in iterables)
        except TypeError:
            length = None

        return FIt(zip_longest(*iterables, fillvalue=fill_value), length)

    #####################
    # itertools recipes #
    #####################

    def take(self, n: int) -> List:
        """Return the first ``n`` items of the iterable as a list

        Cannot be safely used as a static method.
        """
        return list(self.islice(n))

    def tail(self, n: int):
        """Return an iterator over the last ``n`` items"""
        this_len = len_or_none(self)
        if this_len is None:
            length = None
        else:
            length = min(n, this_len)
        return FIt(deque(self, maxlen=n), length)

    def consume(self, n: Optional[int] = None):
        """Advance the iterator ``n`` steps ahead; if ``None``, consume entirely."""
        if n is None:
            deque(self, maxlen=0)
        else:
            next(self.islice(n, n), None)
        return self

    def get(self, n: int, default=None):
        """Alias for ``FIt.nth``: returns the nth item or a default value

        Cannot be safely used as a static method.
        """
        return next(self.islice(n, n + 1), default)

    def nth(self, n: int, default=None):
        """Returns the nth item or a default value

        Cannot be safely used as a static method.
        """
        return self.get(n, default)

    def flatten(self, levels: Optional[int] = None, split_strings: bool = False):
        """Recursively flatten arbitrary iterables (depth-first).

        By default, strings are not treated as iterables to descend into.
        If ``split_strings`` is truthy, their characters will be yielded individually.

        :param levels: How deep in the iterable to flatten (default all levels)
        :param split_strings: Whether to yield individual characters from strings (default False)
        :return: FIt
        """  # noqa
        if levels is None:
            levels = float("inf")

        def gen(obj, lvls):
            if isinstance(obj, str) and (len(obj) == 1 or not split_strings):
                yield obj
                return

            try:
                it = iter(obj)
            except TypeError:
                yield obj
                return

            if lvls <= 0:
                yield from it
            else:
                for item in it:
                    yield from gen(item, lvls - 1)

        this_len = len_or_none(self)
        if levels == 0 and this_len is not None:
            length = this_len
        else:
            length = None

        return FIt(gen(self, levels), length)

    ##########
    # others #
    ##########

    def sliding_window(self, n: int):
        """Iterate over ``n``-length tuples forming a sliding window over the iterable.

        :param n: window size
        :return: FIt of tuples
        """
        this_len = len_or_none(self)
        length = None if this_len is None else max(this_len - n + 1, 0)

        def gen(it):
            window = deque(it.take(n), maxlen=n)
            if len(window) < n:
                return
            yield tuple(window)
            for item in it:
                window.append(item)
                yield tuple(window)

        return FIt(gen(self), length)

    def chunk(self, chunksize: int):
        """Iterate over ``chunksize``-or-shorter lists which are chunks of the iterable.

        :param chunksize: maximum length for each chunk (all but the last chunk will be this size)
        :return: FIt of lists
        """  # noqa
        this_len = len_or_none(self)
        length = None if this_len is None else math.ceil(this_len / chunksize)

        def gen(it):
            it = iter(it)
            taken = []
            while True:
                for _ in range(chunksize):
                    try:
                        taken.append(next(it))
                    except StopIteration:
                        yield taken
                        return
                yield taken
                taken = []

        return FIt(gen(self), length)

    def interleave(self, *iterables):
        """Interleave items from any number of iterables

        When an iterable is exhausted, items continue to be yielded from the remaining
        iterables.

        :param iterables: iterables providing items
        :return:
        """
        iterables = [self] + list(iterables)
        try:
            length = sum(len_or_none(it) for it in iterables)
        except TypeError:
            length = None

        def gen(its):
            its = [iter(it) for it in its]
            while its:
                next_its = []
                for it in its:
                    try:
                        yield next(it)
                        next_its.append(it)
                    except StopIteration:
                        pass

                its = next_its

        return FIt(gen(iterables), length)

    def peek(self, n: Optional[int] = None):
        """Return a list of the next ``n`` items without advancing the iterator.

        If ``n`` is None, return a single item.

        If the iterator is exhausted, the list will be shorter than ``n``; in this case,
        if ``n`` is None, an IndexError will be raised.

        Increments ``consumed``, but does not decrement ``length``.

        Cannot be safely used as a static method.

        :param n:
        :return:
        """
        if n == 0:
            return []

        n_ = n or 1

        peeked = self.take(n_)
        self.cache.extendleft(reversed(peeked))

        return peeked[0] if n is None else peeked

    def for_each(self, function: Callable, *args, **kwargs):
        """Consume the iterator, applying a callable on each item.
        Return values are discarded.

        :param function: callable which takes an iterable as the first argument
        :param args: additional args to pass to the callable
        :param kwargs: additional kwargs to pass to the callable
        :return:
        """
        for item in self:
            function(item, *args, **kwargs)

    ############
    # optional #
    ############

    def progress(self, **kwargs):
        """Create a tqdm progress bar for this iterable.

        :param kwargs: passed to tqdm instance
        :return: FIt wrapping a tqdm instance
        """
        try:
            from tqdm import tqdm

            return FIt(tqdm(self, **kwargs))
        except ImportError:
            warnings.warn("Progress bar is not available: pip install tqdm")
            this_len = len_or_none(self)
            if this_len is None:
                this_len = kwargs.get("total")
            return FIt(self, this_len)
