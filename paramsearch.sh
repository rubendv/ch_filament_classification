#!/usr/bin/env bash

./run.sh python3 paramsearch.py ../data/FORECAST_AIA_HMI_corr.csv \
&& ./run.sh python3 paramsearch.py ../data/SPOCA_DATASET_AIA.csv \
&& ./run.sh python3 paramsearch.py ../data/SPOCA_DATASET_AIA_HMI.csv
