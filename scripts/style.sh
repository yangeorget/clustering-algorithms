#!/bin/bash

echo "Running isort" && \
isort --trailing-comma clustering_algorithms demo.py && \
echo "Running black" && \
black clustering_algorithms demo.py && \
echo "Running flake8" && \
flake8 clustering_algorithms demo.py && \
echo "Running mypy" && \
mypy --disallow-untyped-defs --ignore-missing-imports clustering_algorithms demo.py
