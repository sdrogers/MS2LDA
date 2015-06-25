#!/bin/bash

ks=( 100 200 300 400 500 600 700 800 900 1000)
data='BEER3POS'

rm commands.txt
for k in "${ks[@]}"; do
    echo "./process.sh ${k} ${data}" >> commands.txt
done
