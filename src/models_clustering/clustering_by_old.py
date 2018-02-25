import numpy as np
import pandas as pd
import sys, os

dist_no = sys.argv[1]
dist = "./data/processed/csj/"+dist_no
ext = "./data/external/"
answer_path = ext+"CJLC-0.1/"+dist_no+'.txt'

# ### answerにfreqそれぞれの単語が存在するかのフラグを作る

answer = ""
with open(answer_path) as fh:
    answer = fh.read()

freq = pd.read_csv(dist+'/freqency.csv',header=None)

# general wordsを除外
gw = pd.read_pickle(ext+"/general_words_300.pd")
freq = freq[~freq[0].isin(gw[0])]

freq_flag = freq.apply(lambda word: answer.find(word.values[0]) is not -1, axis=1)

# ### freqに特徴量とベクトルを追加する
freq.columns = ['単語','出現回数','平均信頼度']
freq['正誤'] = freq_flag
freq['A'] = freq['出現回数'] * freq['平均信頼度']
freq['logA'] = freq['A'].apply(lambda x: np.log(x))
freq['sqrtA'] = freq['A'].apply(lambda x: np.sqrt(x))

w2v_all = [
    pd.read_pickle(ext+'/w2v50.pd'),
    pd.read_pickle(ext+'/w2v200.pd')
]

w = w2v_all[0]
#w = w[w.isInDic == True]
w2v_all[0] = w[w.isNWord == True].ix[:,0:50]
w = w2v_all[1]
w2v_all[1] = w[w.isNWord == True].ix[:,0:200]

w2v_all = [w[~w.index.isin(gw[0])] for w in w2v_all]

w2v_freq = [w[w.index.isin(freq['単語'])] for w in w2v_all]


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
import csv

from IPython import embed
from tqdm import tqdm as monitor

# Seed
rng = np.random.RandomState(42)

# データ
n_samples = len(freq)

#--------------------------------------------------

ww_list = w2v_all[0].index
wv_list = w2v_all[0].as_matrix()
w2v_freq[0] = w2v_freq[0][w2v_freq[0].index.isin(freq[freq['A'] > 10]['単語'])] # 10以下を切る
fv_list = w2v_freq[0].as_matrix()
fa_list = freq['A'].values # Aを掛ける(後で)

from numpy import dot

wv_power_list = [dot(wv, wv) for wv in wv_list]
fv_power_list = [dot(fv, fv) for fv in fv_list]

from math import sqrt

w_result = []
for (ww,wv,wv_power) in monitor(zip(ww_list,wv_list,wv_power_list)):
    points = [(dot(wv,fv) / (sqrt(wv_power) * sqrt(fv_power))) * fa for (fv,fa,fv_power) in zip(fv_list,fa_list,fv_power_list)]
    distance = np.mean(points)
    w_result.append([ww,distance])

#embed()

# Save Z
result = pd.DataFrame(w_result)
result = result.sort_values(by=1, ascending=False)
name = (dist+'/clustering-old-')
print('save file. name is "'+name+'"')
result.to_csv(name,header=None,index=False)
