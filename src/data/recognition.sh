#!/bin/bash

# recognition.sh [Number(1-7)]

mkdir "data/processed/csj"

DIST="data/processed/csj/$1"

mkdir $DIST

echo "First recognitoin with Julius..."
echo "data/external/CJLC-0.1/$1.wav" > $DIST/filelist.txt
julius \
    -C lib/dictation-kit/main.jconf \
    -C lib/dictation-kit/am-dnn.jconf \
    -C config/julius_additional.jconf \
    -dnnconf lib/dictation-kit/julius.dnnconf \
    -filelist $DIST/filelist.txt \
    > $DIST/recognition.txt
