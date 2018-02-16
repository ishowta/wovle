#!/bin/bash

# build_features.sh [Number(1-7)]

DIST="data/processed/csj/$1"
RECOGNITION_RESULT_PATH="data/processed/csj/$1/recognition.txt"
FREQENCY_PATH="data/processed/csj/$1/freqency.csv"

echo "recognition result -> freqency data"
nkf -w $RECOGNITION_RESULT_PATH > $RECOGNITION_RESULT_PATH".utf8"
python src/features/extract_sentence.py $RECOGNITION_RESULT_PATH".utf8" $DIST > $FREQENCY_PATH
