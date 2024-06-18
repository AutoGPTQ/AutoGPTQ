#!/bin/bash

pip install -U ruff==0.4.9

ruff check ../auto_gptq_next ../examples ../tests ../setup.py --fix
