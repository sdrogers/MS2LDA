#!/bin/bash

ks=( 50 100 150 200 250 300 350 400 )

rm commands.txt
for k in "${ks[@]}"; do
    echo "./process.sh ${k}" >> commands.txt
done
