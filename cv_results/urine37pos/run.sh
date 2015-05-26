#!/bin/bash

ks=( 25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 )
data='URINE37POS'

rm commands.txt
for k in "${ks[@]}"; do
    echo "./process.sh ${k} ${data}" >> commands.txt
done
