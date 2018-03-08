import sys
import os


def flatten(list):
    return [flatten for inner in list for flatten in inner]


input_file_path = sys.argv[1]
dist_path = sys.argv[2]

# 認識結果から単語列と信頼度列のリストを作る
# 最初候補文字列が30個出る設定でやっていたのでよくわからないコードになっている
sentences_list = []
scores_list = []
with open(input_file_path) as f:
    text = f.readlines()
    text = text[text.index('STAT: ### speech analysis (waveform -> MFCC)\n'):]
    sentences_list = list(filter(lambda s: 'sentence1:' in s, text))
    scores_list = list(filter(lambda s: 'cmscore1:' in s, text))
size = len(sentences_list)
word_dict = []
for st in range(0, size, 1):
    sentence = sentences_list[st:st+1]
    score = scores_list[st:st+1]
    words = flatten([s.split(' ')[1:] for s in sentence])
    words = [s.replace('\n', '') for s in words]
    word_scores = flatten([s.split(' ')[1:] for s in score])
    word_scores = [s.replace('\n', '') for s in word_scores]

    for i, word in enumerate(words):
        if word not in word_dict:
            word_dict.append([word, word_scores[i]])

# 重複除去、信頼度は合計値を出す
unique_word_dict = {}
for word in word_dict:
    if word[0] not in unique_word_dict:
        unique_word_dict[word[0]] = [1, float(word[1])]
    else:
        [cnt, v] = unique_word_dict[word[0]]
        unique_word_dict[word[0]] = [cnt+1, v+float(word[1])]

# 単語だけファイルに書き込んでmecabで名詞だけ取り出し、
# mecabによって除去された単語をリストから除去
word_list_path = dist_path+"/recognition_wordlist.csv"
mecab_word_list_path = dist_path+"/recognition_wordlist_mecab.csv"
with open(word_list_path, "w") as f:
    for k, v in unique_word_dict.items():
        f.write(k+"\n")
os.system("src/features/text_to_words.sh "
          + word_list_path
          + " > "
          + mecab_word_list_path)
new_unique_dict = {}
with open(mecab_word_list_path) as f:
    word_list = f.readlines()
    word_list = [s.replace('\n', '') for s in word_list]
    for k, v in unique_word_dict.items():
        if k in word_list:
            new_unique_dict[k] = v

# 単語と重要度を出力
for k, v in new_unique_dict.items():
    if not int(v[0]) == 1:
        print(k+","+str(v[0])+","+str(float(v[1])/float(v[0])))  # 信頼度の合計を割ってる
