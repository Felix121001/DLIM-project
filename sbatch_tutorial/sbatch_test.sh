#!/bin/bash
source /usr/itetnas04/data-scratch-01/dlim_13hs23/data/conda/bin/conda/bin/conda shell.bash hook
conda activate
python -u test.py "$@"
