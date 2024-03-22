#!/bin/bash

echo "Running isort" && \
isort --trailing-comma clustering && \
echo "Running black" && \
black clustering && \
echo "Running flake8" && \
flake8 clustering && \
echo "Running mypy" && \
mypy --disallow-untyped-defs --ignore-missing-imports clustering
