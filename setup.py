import io
import os
import re

from setuptools import find_packages
from setuptools import setup

__version__ = '0.19.0'

def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())

setup(
    name="eepackages",
    version=__version__,
    url="https://github.com/gee-community/ee-packages-py",
    license='MIT',

    author="Gennadii Donchyts",
    author_email="gennadiy.donchyts@gmail.com",

    description="A set of utilities built on top of Google Earth Engine (migrated from JavaScript)",
    long_description=read("README.md"),

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        "earthengine-api>=0.1.284",
        "pathos>=0.2.8",
        "retry>=0.9.2"
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
