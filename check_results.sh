#!/bin/bash
printf "model & splits & degree & mEvar & mMAE & mMSE & mMax & mR2 & slEvar & slMAE & slMSE & slMax & slR2\n"
for i in $(ls ./mlruns/0/); do
	if [ -d ./mlruns/0/$i ]; then
		#printf " & "
		cat mlruns/0/$i/params/model
		printf " & "
		cat mlruns/0/$i/params/splits
		printf " & "
		cat mlruns/0/$i/params/degree
		for j in $(ls ./mlruns/0/${i}/metrics/); do
			printf " & "
			awk '{printf "%.2f",$2}' mlruns/0/$i/metrics/$j
		done
		printf "\n"
	fi
done
