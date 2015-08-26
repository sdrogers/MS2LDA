#!/bin/bash

K=$1
DATA=$2

export PYTHONPATH="../../.."
python ../../lda_cv.py $K $DATA
