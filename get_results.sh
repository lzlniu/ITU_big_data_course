#!/bin/bash

for i in `seq 2 7`; do
	for j in `seq 2 7`; do
		mlflow run . -P splits=$i -P degree=$j -P model='LinReg'
		mlflow run . -P splits=$i -P degree=$j -P model='RFReg'
		mlflow run . -P splits=$i -P degree=$j -P model='SVR'
	done
done

./check_results.sh > results.tex
./check_results.sh | sed -e 's/ & /\t/g' > results.tsv

cat results.tsv

