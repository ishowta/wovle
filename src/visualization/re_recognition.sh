#!/bin/bash

# re_recognition.sh [Number(1-7)]

DIST="data/processed/csj/$1"
NAME="model-$2-clustering-50-10-logA-SVM-0.25.sjis"


echo "-v $NAME" > $DIST"/updated-"$NAME".jconf"

echo "Re recognitoin with Julius..."
lib/julius \
    -C lib/ssr-kit-v4.4.2.1a/main.jconf \
    -C config/julius_additional.jconf \
    -C $DIST"/updated-"$NAME".jconf" \
    -dnnconf config/julius_dnn.dnnconf \
    -filelist $DIST/filelist.txt \
    > $DIST"/re_recognition-"$NAME".txt"
