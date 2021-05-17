from itertools import chain
from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()

extras = {"progress": ["tqdm"]}
extras["all"] = sorted(set(chain.from_iterable(extras.values())))

setup(
    name="f_it",
    url="https://github.com/clbarnes/f_it",
    author="Chris L. Barnes",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["f_it*"]),
    install_requires=[],
    python_requires=">=3.7, <4.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    extras_require=extras,
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
