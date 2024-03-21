#!/bin/bash

echo "Running isort" && \
isort --trailing-comma *.py && \
echo "Running black" && \
black *.py && \
echo "Running flake8" && \
flake8 *.py && \
echo "Running mypy" && \
mypy --disallow-untyped-defs --ignore-missing-imports *.py
