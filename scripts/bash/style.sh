#!/bin/bash

echo "Running isort" && \
isort --trailing-comma clustering_algorithms scripts/python && \
echo "Running black" && \
black clustering_algorithms scripts/python && \
echo "Running flake8" && \
flake8 clustering_algorithms scripts/python && \
echo "Running mypy" && \
mypy --disallow-untyped-defs --ignore-missing-imports clustering_algorithms scripts/python
