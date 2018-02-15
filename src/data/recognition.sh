#!/bin/bash

# recognition.sh [Number(1-7)]

mkdir "data/processed/csj"

DIST="data/processed/csj/$1"

mkdir $DIST

echo "First recognitoin with Julius..."
echo "data/external/CJLC-0.1/$1.wav" > $DIST/filelist.txt
lib/julius \
    -C lib/ssr-kit-v4.4.2.1a/main.jconf \
    -C config/julius_additional.jconf \
    -dnnconf config/julius_dnn.dnnconf \
    -filelist $DIST/filelist.txt \
    > $DIST/recognition.txt
