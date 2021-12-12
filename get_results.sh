#!/bin/bash

for i in `seq 3 20`; do
	mlflow run . -P degree=$i
done

./check_results.sh > results.tex
./check_results.sh | sed -e 's/ & /\t/g' > results.tsv

cat results.tsv

