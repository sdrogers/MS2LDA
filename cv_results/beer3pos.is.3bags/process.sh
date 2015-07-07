#!/bin/bash

K=$1
DATA=$2

PYTHONPATH=../../../ python ../../experimental/lda_3bags_cv.py $K $DATA
