# f_it

Iterator class for functional-esque, postfix-chaining programming.

## Features

* A single wrapper class exposing chain-able methods for lazily transforming iterators
* Wraps functions from `functools`, `itertools`, and some extras
* Optionally has a length, which is calculated for subsequent operations if possible

Note that this package is for convenience/ interface comfort purposes
and does not provide the guarantees of a true functional language.
There may be a significant performance overhead to using deeply nested ``FIt`` instances in tight loops.

## Usage

```python
from f_it import FIt

it = FIt(range(10))
transformed = it.map(  # cube elements
    lambda x: x**3
).filter(  # drop even elements
    lambda x: x % 2
).cycle(  # repeat the whole iterator 3 times
    3
).islice(  # take some elements from the middle
    5, 10
).chain(  # add 0-4 to the end
    range(5)
).chunk(  # separate into 2-length chunks
    2
)

# __add__ and __radd__ are implemented for chaining other Iterators
added = transformed + iter([1, 2, 3])

# nothing has been evaluated yet!

# evaluate operations, reading into a list
# if tqdm is available, show progress bar
as_list = added.progress().to(list)
```
