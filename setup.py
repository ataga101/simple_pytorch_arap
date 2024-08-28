# Author: Kenta Nakamura <c60evaporator@gmail.com>
# Copyright (c) 2020-2021 Kenta Nakamura
# License: BSD 3 clause

from setuptools import setup

DESCRIPTION = "pytorch As-Rigid-As-Possible loss function"
NAME = 'pytorch_arap_simple'
VERSION = "0.1.0"
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'torch>=1.6.0',
    'numpy>=1.18.5',
]

setup(name=NAME,
      description=DESCRIPTION,
      version=VERSION,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
)
