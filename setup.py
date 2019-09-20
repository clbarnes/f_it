#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os

from setuptools import setup, find_packages
import itertools

here = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(here, "README.rst")) as readme_file:
    readme = readme_file.read()

with open(os.path.join(here, "HISTORY.rst")) as history_file:
    history = history_file.read()

requirements = []

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest"]

extras_require = {}
extras_require["all"] = list(itertools.chain.from_iterable(extras_require.values()))

setup(
    author="Chris L. Barnes",
    author_email="chrislloydbarnes@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Iterator class for functional programming",
    install_requires=requirements,
    extras_require=extras_require,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="f_it",
    name="f_it",
    packages=find_packages(include=["f_it"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/clbarnes/f_it",
    version="0.2.1",
    zip_safe=False,
)
