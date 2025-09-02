#!/bin/bash

dataset=$1

python scripts/llm_filter.py -i rules/${dataset}/${dataset}_llm.pl -l raw
python scripts/llm_filter.py -i rules/${dataset}/${dataset}_llm.txt -l interpreted
python scripts/llm_filter.py -i rules/${dataset}/${dataset}_llm.txt.en -l english