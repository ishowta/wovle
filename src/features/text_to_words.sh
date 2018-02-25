#!/bin/sh

mecab -d `mecab-config --dicdir`"/mecab-ipadic-neologd" $1 \
| grep -e "名詞,一般" -e "名詞,サ変接続" -e "名詞,固有名詞" \
| grep -v "代 名詞" \
| grep -v "名詞,代名詞" \
| grep -v "もの" \
| grep -v "あと" \
| grep -v "わけ" \
| grep -v "とこ" \
| grep -v "ば" \
| grep -v "ところ" \
| grep -v "たい" \
| cut -f1 \
| sort
