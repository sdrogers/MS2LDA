#!/bin/bash

ks=( 100 120 140 160 180 200 220 240 260 280 300 320 340 )

rm commands.txt
for k in "${ks[@]}"; do
    echo "./process.sh ${k}" >> commands.txt
done
