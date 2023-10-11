#!/bin/bash
set -e

# Upgrade pip
python -m pip install --upgrade pip

# Install pycodestyle
pip install mypy
pip install types-toml

mypy --config-file ./mypi.ini RecurrentFF/model/

# TODO: clean these up
# mypy --config-file ./mypi.ini RecurrentFF/benchmarks/

echo "Mypy check passed successfully!"