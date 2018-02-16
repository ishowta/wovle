import sys
import os
from pprint import pprint

def flatten(list):
    return [flatten for inner in list for flatten in inner]

input_file_path = sys.argv[1]
dist_path = sys.argv[2]
#output_file_path = sys.argv[3]

# 単語列と信頼度列のリストを作る
sentences_list = []
scores_list = []
with open(input_file_path) as f:
    text = f.readlines()
    text = text[text.index('STAT: ### speech analysis (waveform -> MFCC)\n'):]
    sentences_list = list(filter(lambda s: 'sentence1:' in s, text))
    scores_list = list(filter(lambda s: 'cmscore1:' in s, text))

size = len(sentences_list)
word_dict = []
for st in range(0, size, 1): #30
    sentence = sentences_list[st:st+1] #30
    score = scores_list[st:st+1] #30
    words = flatten([ s.split(' ')[1:] for s in sentence])
    words = [s.replace('\n','') for s in words]
    word_scores = flatten([ s.split(' ')[1:] for s in score])
    word_scores = [s.replace('\n','') for s in word_scores]


    for i, word in enumerate(words):
        if not word in word_dict:
            word_dict.append([word, word_scores[i]])
    #word_dict.sort(key=lambda x:x[0])

# erase duplicate
unique_word_dict = {}
for word in word_dict:
    if word[0] not in unique_word_dict:
        unique_word_dict[word[0]] = [1,float(word[1])]
    else:
        [cnt, v] = unique_word_dict[word[0]]
        unique_word_dict[word[0]] = [cnt+1,v+float(word[1])]

# 単語だけ書き込み
word_list_path = dist_path+"/recognition_wordlist.csv"
with open(word_list_path, "w") as f:
    for k,v in unique_word_dict.items():
        f.write(k+"\n")

# mecab
os.system("src/features/text_to_words.sh "+word_list_path+" > "+dist_path+"/recognition_wordlist_mecab.txt")

# mecabによって除去された単語を除去
new_unique_dict = {}
with open(dist_path+"/recognition_wordlist_mecab.txt") as f:
    word_list = f.readlines()
    word_list = [ s.replace('\n','') for s in word_list]
    for k,v in unique_word_dict.items():
        if k in word_list:
            new_unique_dict[k] = v

for k,v in new_unique_dict.items():
    if not int(v[0]) == 1:
        # and not int(v[0]) == 2:
        print(k+","+str(v[0])+","+str(float(v[1])/float(v[0])))
