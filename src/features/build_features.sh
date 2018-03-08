#!/bin/bash

# build_features.sh [Number(1-7)]

DIST="data/processed/corpus/$1"
RECOGNITION_RESULT_PATH="data/processed/corpus/$1/recognition.txt"
FREQENCY_PATH="data/processed/corpus/$1/freqency.csv"

echo "recognition result -> freqency data"
python src/features/extract_sentence.py $RECOGNITION_RESULT_PATH $DIST > $FREQENCY_PATH
