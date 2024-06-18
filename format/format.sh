#!/bin/bash

# force ruff/isort to be same version as setup.py
pip install -U ruff==0.4.9 isort==5.13.2

isort -l 119 -e ../
ruff check ../auto_gptq_next ../examples ../tests ../setup.py --fix
