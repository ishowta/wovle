#!/bin/bash

# re_recognition.sh [Number(1-7)]

DIST="data/processed/corpus/$1"
NAME="model-clustering-$2"

echo "-v $NAME" > $DIST"/updated-"$NAME".jconf"

echo "Re recognitoin with Julius..."
julius \
    -C lib/dictation-kit/main.jconf \
    -C lib/dictation-kit/am-dnn.jconf \
    -C config/julius_additional.jconf \
    -C $DIST"/updated-"$NAME".jconf" \
    -dnnconf lib/dictation-kit/julius.dnnconf \
    -filelist $DIST/filelist.txt \
    > $DIST"/re_recognition-"$NAME".txt"
