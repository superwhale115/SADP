import os

from setuptools import setup, find_packages

import gops


def find_data_packages(where):
    data_packages = []
    for filepath, dirnames, filenames in os.walk(where):
        filepath = filepath.replace('\\', '/') + '/*'
        filepath = filepath[len(where) + 1:]
        data_packages.append(filepath)
    return data_packages


setup(
    name='gops',
    version=gops.__version__,
    description='General Optimal control Problem Solver (GOPS)',
    packages=[package for package in find_packages() if package.startswith('gops')],
    package_data={
        'gops.env': find_data_packages('gops/env')
    },
    install_requires=[
        'torch>=1.6.0',
        'numpy>1.16.0',
        'ray>=1.0.0',
        'gym==0.23.1',
        'pygame',
        'box2d',
        'pandas',
        'tensorboard>=2.5.0',
        'matplotlib',
        'pyglet',
        'seaborn',
        'scipy',
        'slxpy',
        'openpyxl'
    ],
    python_requires='>=3.6',
)