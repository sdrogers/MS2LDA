#!/bin/bash

ks=( 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
data='BEER1POS'

rm commands.txt
for k in "${ks[@]}"; do
    echo "./process.sh ${k} ${data}" >> commands.txt
done