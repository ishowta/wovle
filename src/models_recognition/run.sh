#!/bin/bash

# $1 ... データ番号[1-7]
# $2 ... モード[1-3]

DIST="./data/processed/csj/$1"
LANG_MODEL_PATH="./data/raw/hybrid.htkdic" #utf8?

CONFIG_NAME="clustering-50-10-logA-SVM-0.25"
MODE=$2

DISTANCE_PATH=$DIST/$CONFIG_NAME
UPDATED_LANG_MODEL_PATH=$DIST/"model-"$MODE"-"$CONFIG_NAME

# 5. 言語モデルの更新
echo "Update language model"
# コンパイルは一回だけでいい…
g++ -std=c++17 -Wall -O2 src/models_recognition/update_langmodel.cpp -o ./data/processed/update_langmodel
./data/processed/update_langmodel $DISTANCE_PATH $LANG_MODEL_PATH $UPDATED_LANG_MODEL_PATH $DIST $MODE
nkf -s $UPDATED_LANG_MODEL_PATH > $UPDATED_LANG_MODEL_PATH.sjis