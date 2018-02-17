import sys
import os
from pprint import pprint

input_file_path = sys.argv[1]

sentences_list = []
with open(input_file_path) as f:
    text = f.readlines()
    text = text[text.index('STAT: ### speech analysis (waveform -> MFCC)\n'):]
    sentences_list = list(filter(lambda s: 'sentence1:' in s, text))

for sentence in sentences_list:
    print(sentence
        .replace('sentence1:','')
        .replace('\n','')
        .replace(' ',''))