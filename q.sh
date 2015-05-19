#!/bin/bash

rm commands.txt 2>/dev/null
./run.sh
lines=$(wc -l < commands.txt)
echo "${lines} lines in commands.txt"
qsub -S /bin/bash -V -cwd -j y -N LDA -t 1:${lines} script_wrapper.sh

qstat
read -rsp $'Press enter to continue...\n'
