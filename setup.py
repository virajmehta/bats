"""
Setup file.

Author: Ian Char
Date: 8/26/2020
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bats",
    version="0.0.1",
    author="Viraj Mehta, Ian Char",
    author_email="virajm@cs.cmu.edu, ichar@cs.cmu.edu",
    description="BATS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/virajmehta/bats",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
