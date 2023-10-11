#!/bin/bash
set -e

# Upgrade pip
python -m pip install --upgrade pip

# Install pycodestyle
pip install mypy

mypy --config-file ./mypi.ini RecurrentFF/model/

# TODO: clean these up
# mypy --config-file ./mypi.ini RecurrentFF/benchmarks/

echo "Mypy check passed successfully!"