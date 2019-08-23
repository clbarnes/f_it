=========================
f_it: Functional Iterator
=========================


.. image:: https://img.shields.io/pypi/pyversions/f_it.svg
        :target: https://pypi.python.org/pypi/f_it

.. image:: https://img.shields.io/pypi/v/f_it.svg
        :target: https://pypi.python.org/pypi/f_it

.. image:: https://img.shields.io/travis/clbarnes/f_it.svg
        :target: https://travis-ci.org/clbarnes/f_it

.. image:: https://readthedocs.org/projects/f_it/badge/?version=latest
        :target: https://f_it.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black

Iterator class for functional programming in python

* Free software: MIT license
* Documentation: https://f_it.readthedocs.io.

Features
--------

* A single wrapper class exposing chain-able methods for lazily transforming iterators
* Wraps functions from ``functools``, ``itertools``, and some extras
* Optionally has a length, which is calculated for subsequent operations if possible

Note that this package is for convenience/ interface comfort purposes
and does not provide the guarantees of a true functional language.

Install
-------

``pip install f_it``

Usage
-----

.. code-block:: python

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

    # nothing has been evaluated yet!

    # evaluate operations, reading into a list
    # if tqdm is available, show progress bar
    as_list = transformed.progress().to(list)
