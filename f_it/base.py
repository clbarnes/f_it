# -*- coding: utf-8 -*-
import math
import warnings
from abc import ABC, abstractmethod
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


class BaseFIt(ABC):
    _iter = None

    def __init__(self, iterable, length=None):
        """Iterator class providing many postfix functional methods.

        Most of these methods can also be used as static methods which take any
        iterable as the first argument.
        Where possible, the returned FIt instances have a length, which expresses how
        many items *remain* in the iterator.

        Where possible, all iteration is evaluated lazily.

        :param iterable: iterable to wrap
        :param length: explicitly provide a length if you know it but the iterable doesn't
        """  # noqa
        self.iterator = self._iter(iterable)
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

    @abstractmethod
    def next(self):
        """Return the next item in the iterator"""

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

    @abstractmethod
    def enumerate(self, start=0):
        """See the builtin enumerate_

        .. _enumerate: https://docs.python.org/3/library/functions.html#enumerate
        """

    @abstractmethod
    def filter(self, function=None):
        """See the builtin filter_

        .. _filter: https://docs.python.org/3/library/functions.html#filter
        """

    @abstractmethod
    def map(self, function, *iterables, longest=True):
        """See the builtin map_

        Difference from stdlib: accepts ``longest`` kwarg, for whether to terminate
        with the  longest iterable (padding the others with ``None``) instead of
        terminating with the shortest.

        .. _map: https://docs.python.org/3/library/functions.html#map
        """

    @classmethod
    @abstractmethod
    def range(cls, start, stop=None, step=1):
        """See the builtin range_

        .. _range: https://docs.python.org/3/library/functions.html#func-range
        """

    @abstractmethod
    def zip(self, *iterables, longest=False, fill_value=None):
        """See the builtin zip_

        Difference from stdlib: accepts ``longest`` kwarg, which makes this method
        act like itertools.zip_longest_ (also accepts ``fill_value``)

        .. _zip: https://docs.python.org/3/library/functions.html#zip
        .. _zip_longest: https://docs.python.org/3/library/itertools.html#itertools.zip_longest
        """  # noqa

    #############
    # functools #
    #############

    @abstractmethod
    def reduce(self, function, initializer=None):
        """See functools.reduce_

        Difference from stdlib: if ``initializer`` is ``None``, method will behave as if
        it was not given.

        .. _functools.reduce: https://docs.python.org/3/library/functools.html#functools.reduce
        """  # noqa

    #############
    # itertools #
    #############

    @abstractmethod
    def chain(self, *iterables):
        """See itertools.chain_

        .. _itertools.chain: https://docs.python.org/3/library/itertools.html#itertools.chain
        """  # noqa

    @abstractmethod
    def chain_from_iterable(self, iterable):
        """See itertools.chain.from_iterable_

        .. _itertools.chain.from_iterable: https://docs.python.org/3/library/itertools.html#itertools.chain.from_iterable
        """  # noqa

    @abstractmethod
    def combinations(self, r: int, replace=False):
        """See itertools.combinations_

        Difference from stdlib: ``replace`` argument to use
        itertools.combinations_with_replacement_

        .. _itertools.combinations: https://docs.python.org/3/library/itertools.html#itertools.combinations
        .. _itertools.combinations_with_replacement: https://docs.python.org/3/library/itertools.html#itertools.combinations_with_replacement
        """  # noqa

    @abstractmethod
    def combinations_with_replacement(self, r: int):
        """See itertools.combinations_with_replacement_

        .. _itertools.combinations_with_replacement: https://docs.python.org/3/library/itertools.html#itertools.combinations_with_replacement
        """  # noqa

    @abstractmethod
    def compress(self, selectors: Iterable):
        """See itertools.compress_

        .. _itertools.compress: https://docs.python.org/3/library/itertools.html#itertools.compress
        """  # noqa

    @classmethod
    @abstractmethod
    def count(cls, start: int = 0, step: int = 1):
        """See itertools.count_

        .. _itertools.count: https://docs.python.org/3/library/itertools.html#itertools.count
        """  # noqa

    @abstractmethod
    def cycle(self, n: Optional[int] = None):
        """See itertools.cycle_

        Difference from stdlib: accepts ``n`` argument, for how many times it should be
        repeated.

        .. _itertools.cycle: https://docs.python.org/3/library/itertools.html#itertools.cycle
        """  # noqa

    @abstractmethod
    def dropwhile(self, predicate: Optional[Callable[[Any], bool]] = None):
        """See itertools.dropwhile_

        Difference from stdlib: if predicate is None (default), use ``bool``

        .. _itertools.dropwhile: https://docs.python.org/3/library/itertools.html#itertools.dropwhile
        """  # noqa

    @abstractmethod
    def filterfalse(self, predicate: Callable[[Any], bool] = None):
        """See itertools.filterfalse_

        Difference from stdlib: if predicate is None (default), use ``bool``

        .. _itertools.filterfalse: https://docs.python.org/3/library/itertools.html#itertools.filterfalse
        """  # noqa

    @abstractmethod
    def groupby(self, key: Optional[Callable[[Any], Any]] = None):
        """See itertools.groupby_

        .. _itertools.groupby: https://docs.python.org/3/library/itertools.html#itertools.groupby
        """  # noqa

    @abstractmethod
    def islice(self, start: int, stop: Optional[int] = None, step: int = 1):
        """See itertools.islice_

        .. _itertools.islice: https://docs.python.org/3/library/itertools.html#itertools.islice
        """  # noqa

    @abstractmethod
    def permutations(self, r: Optional[int] = None):
        """See itertools.permutations_

        .. _itertools.permutations: https://docs.python.org/3/library/itertools.html#itertools.permutations
        """  # noqa

    @abstractmethod
    def product(self, *iterables, repeat: int = 1):
        """See itertools.product_

        .. _itertools.product: https://docs.python.org/3/library/itertools.html#itertools.product
        """  # noqa

    @classmethod
    @abstractmethod
    def repeat(cls, obj, n: Optional[int] = None):
        """See itertools.repeat_

        .. _itertools.repeat: https://docs.python.org/3/library/itertools.html#itertools.repeat
        """  # noqa

    @abstractmethod
    def starmap(self, function: Callable):
        """See itertools.starmap_

        .. _itertools.starmap: https://docs.python.org/3/library/itertools.html#itertools.starmap
        """  # noqa

    @abstractmethod
    def takewhile(self, predicate: Optional[Callable[[Any], bool]] = None):
        """See itertools.takewhile_

        .. _itertools.takewhile: https://docs.python.org/3/library/itertools.html#itertools.takewhile
        """  # noqa

    @abstractmethod
    def tee(self, n: int = 2):
        """See itertools.tee_

        .. _itertools.tee: https://docs.python.org/3/library/itertools.html#itertools.tee
        """  # noqa

    @abstractmethod
    def zip_longest(self, *iterables, fill_value=None):
        """See itertools.zip_longest_

        .. _itertools.zip_longest: https://docs.python.org/3/library/itertools.html#itertools.zip_longest
        """  # noqa

    #####################
    # itertools recipes #
    #####################

    @abstractmethod
    def take(self, n: int) -> List:
        """Return the first ``n`` items of the iterable as a list

        Cannot be safely used as a static method.
        """

    @abstractmethod
    def tail(self, n: int):
        """Return an iterator over the last ``n`` items"""

    @abstractmethod
    def consume(self, n: Optional[int] = None):
        """Advance the iterator ``n`` steps ahead; if ``None``, consume entirely."""

    @abstractmethod
    def get(self, n: int, default=None):
        """Alias for ``FIt.nth``: returns the nth item or a default value

        Cannot be safely used as a static method.
        """

    @abstractmethod
    def nth(self, n: int, default=None):
        """Returns the nth item or a default value

        Cannot be safely used as a static method.
        """

    @abstractmethod
    def flatten(self, levels: Optional[int] = None, split_strings: bool = False):
        """Recursively flatten arbitrary iterables (depth-first).

        By default, strings are not treated as iterables to descend into.
        If ``split_strings`` is truthy, their characters will be yielded individually.

        :param levels: How deep in the iterable to flatten (default all levels)
        :param split_strings: Whether to yield individual characters from strings (default False)
        :return: FIt
        """  # noqa

    ##########
    # others #
    ##########

    @abstractmethod
    def sliding_window(self, n: int):
        """Iterate over ``n``-length tuples forming a sliding window over the iterable.

        :param n: window size
        :return: FIt of tuples
        """

    @abstractmethod
    def chunk(self, chunksize: int):
        """Iterate over ``chunksize``-or-shorter lists which are chunks of the iterable.

        :param chunksize: maximum length for each chunk (all but the last chunk will be this size)
        :return: FIt of lists
        """  # noqa

    @abstractmethod
    def interleave(self, *iterables):
        """Interleave items from any number of iterables

        When an iterable is exhausted, items continue to be yielded from the remaining
        iterables.

        :param iterables: iterables providing items
        :return:
        """

    @abstractmethod
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

    @abstractmethod
    def for_each(self, function: Callable, *args, **kwargs):
        """Consume the iterator, applying a callable on each item.
        Return values are discarded.

        :param function: callable which takes an iterable as the first argument
        :param args: additional args to pass to the callable
        :param kwargs: additional kwargs to pass to the callable
        :return:
        """

    ############
    # optional #
    ############

    @abstractmethod
    def progress(self, **kwargs):
        """Create a tqdm progress bar for this iterable.

        :param kwargs: passed to tqdm instance
        :return: FIt wrapping a tqdm instance
        """
