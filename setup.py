#!/usr/bin/env python

import re
from setuptools import setup

# parse version from init.py
with open("ipcoal/__init__.py") as init:
    CUR_VERSION = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        init.read(),
        re.M,
    ).group(1)

# setup installation
setup(
    name="ipcoal",
    packages=["ipcoal"],
    version=CUR_VERSION,
    author="Patrick McKenzie and Deren Eaton",
    author_email="p.mckenzie@columbia.edu",
    install_requires=[
        "numpy>=1.9",
        "pandas>=1.0",
        "toytree>=1.1.0",
        "msprime",
        "numba",
        "scipy>0.10",
        # seq-gen (optional)
        # "ipyparallel",
    ],
    license='GPL',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
)
