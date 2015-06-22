#!/bin/bash

ks=( 200 400 600 800 1000 1200 1400 1600 1800 2000 )
data='URINE37POS'

rm commands.txt
for k in "${ks[@]}"; do
    echo "./process.sh ${k} ${data}" >> commands.txt
done
