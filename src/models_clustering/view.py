from IPython import embed
import numpy as np
import pandas as pd
import sys, os
import seaborn as sea


#https://gist.github.com/yumatsuoka/c6402870493adc89b881a0fd263c3628
def calc_anv(hist):
    """
    引数：ヒストグラム(1次元のnumpy.ndarray)
    返値：double型の平均値，ピクセル数，平均値を格納したディクショナリ
    """
    # ヒストグラムの頻度×輝度
    ave = np.mean([i * x for i, x in enumerate(hist)])
    # ヒストグラム内のピクセル数
    n = hist.size
    # ヒストグラムの分散の値
    var = np.sum((hist - ave) ** 2.) / n
    return {'ave':ave, 'n':n,'var':var}
def bet_wit_cv(b_d, w_d):
    """
    引数：２つのディクショナリ
    返値：double型のクラス間分散，クラス内分散の値のタプル
    """
    bc_var = b_d['n'] * w_d['n'] * ((b_d['ave'] - w_d['ave']) ** 2) / ((b_d['n'] + w_d['n']) ** 2)
    wc_var = (b_d['n'] * b_d['var'] + w_d['n'] * w_d['var'] ) / (b_d['n'] + w_d['n'])
    return bc_var, wc_var
def calc_all(b_a, w_a):
    """
    引数：２つのnumpy.ndarray
    返値：double型の分離度
    """
    bc_var, wc_var = bet_wit_cv(b_d=calc_anv(b_a), w_d=calc_anv(w_a))
    t_var = bc_var + wc_var
    s_m = bc_var / (t_var - bc_var)
    return s_m

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

# Seed
rng = np.random.RandomState(42)

# データ
n_samples = len(freq)
# ## パラメータ
# - One-Class SVM
#     - nu
#         - 0.001, 0.05, 0.1, 0.25
# - Isolation Forest
#     - outliners_fraction
#         - 0.05,0.2,0.5
# - LOF
#     - outliners_fraction
#         - 0.05,0.2,0.5
#     - n_neighbors
#         - 5, 35, 70
# - Word2Vec Model
#     - 50 or 200
# - duplicate(n)
#     - そのまま (n)
#     - 頻度による変化 n' = (n*round(A))
#     - log(n')
#     - sqrt(n')
# - 出現回数10?回以上
# 
# = ( 4 + 3 + 3\*3 ) \* 2 \* 4 \* 2  
# = 256 pattern

from itertools import product

params = [
    [0, 1],                         # わからない
    ['n', 'A', 'logA', 'sqrtA'],    # わからない
    [0, 5, 10],                        # 5-10あたりがいい？
    ['SVM', 'IF', 'LOF'],           # SVM or LOF?
]
model_params = {
    'SVM': {'nu':[0.001, 0.05, 0.1, 0.25]}, # 0.1, 0.25が良い
    'IF' : {'of':[0.05,0.2,0.5]},           #
    'LOF': {'of':[0.05,0.2,0.5],            # 0.5   0.5
            'nn':[5,35,70]}                 # 5     35
}


param_list = [
    (0,'n',5,'LOF'),
    (0,'logA',5,'LOF'),
    (0,'sqrtA',5,'LOF'),
    (0,'A',10,'SVM'),
]

model_param_list = {
    'SVM': [[0.1,0.25]],
    'IF' : [[]],
    'LOF': [[0.05,0.5],[5,35]],
}

ap = pd.read_pickle('./pnpd.pd')
ap = ap.values.tolist()
ap.reverse()

print("DISP START")

from matplotlib import pyplot as plt

cnt = 0
scnt = 1
fig = plt.figure()
for param in ap:
    #if param not in param_list:
    #    continue
    w2v = w2v_freq[0 if param[0] == '50' else 1]
    #w2v_a = w2v_all[param[0]]
    w2v_param = 0 if param[0] == '50' else 1
    a_type = param[2]
    count_threshold = int(param[1])
    clustering_model_type = param[3]

    model_param = [float(param[4]), int(param[5]) if param[5] != None else None]

    if scnt == 5:
        fig.canvas.set_window_title(
            'cnt='+str(cnt)
            #+' model='+clustering_model_type
            #+' w2v='+('50' if w2v_param == 0 else '200')
            #+' type='+a_type
            #+' threshold='+str(count_threshold)
            #+' p1='+str(model_param[0])
            #+' p2='+str("none" if len(model_param) == 1 else model_param[1])
        )
        figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        #plt.tight_layout()
        plt.show()
        scnt = 1
        fig = plt.figure()
        cnt += 1
        if cnt > 0 : break

#    if model_param[0] not in model_param_list[clustering_model_type][0]:
#        continue
#    if clustering_model_type == 'LOF' and model_param[1] not in model_param_list[clustering_model_type][1]:
#        continue
    #X = w2v[w2v.index.isin(freq[freq['A'] > count_threshold]['単語'])]
    #Z = w2v[w2v.index.isin(freq[(freq['A'] > 10) | (freq['正誤'] == False)]['単語'])]
    C = w2v[w2v.index.isin(freq[freq['A']     > count_threshold  ]['単語'])]
    V = w2v[w2v.index.isin(freq[freq['A']     > 10             ]['単語'])]
    F = w2v[w2v.index.isin(freq[freq['正誤'] == False           ]['単語'])]
    msk = np.random.rand(len(V)) < 0.6
    trainV = V[msk]
    testV = V[~msk]
    X = C[~C.index.isin(trainV.index)] #X is train
    Z = testV.append(F) #Z is test

    print("param=")
    print(param)

    if a_type != 'n':
        X_cnt = [ int(np.round(freq[freq['単語'] == word][a_type])) for word in X.index]
        buf = []
        for (i,c) in zip(range(len(X)),X_cnt):
            for ci in range(c):
                buf.append(X.ix[[i],:].values[0])
        X = np.array(buf)

    if clustering_model_type == 'SVM':
        model = svm.OneClassSVM(nu=model_param[0], kernel="rbf", gamma="auto")
        model.fit(X)
        score = model.decision_function(Z)
        score = [s[0] for s in score]
    elif clustering_model_type == 'IF':
        model = IsolationForest(max_samples=n_samples, contamination=model_param[0], random_state=rng)
        model.fit(X)
        score = model.decision_function(Z)
    elif clustering_model_type == 'LOF':
        model = LocalOutlierFactor(n_neighbors=model_param[1], contamination=model_param[0])
        model.fit_predict(X)
        score = model._decision_function(Z)

    ax = fig.add_subplot(2,2,scnt)

    yg = np.array([freq[freq['単語'] == word]['正誤'].values[0] for word in Z.index]) # z flag
    xg = np.array(score) # z
    #ax.hist(xg[~yg],normed=True)
    #ax.hist(xg[yg],normed=True)
    sea.distplot(xg[~yg], kde=True, rug=True)
    sea.distplot(xg[yg], kde=True, rug=True)
    #ax.scatter(xg,yg)
    ax.grid(True)
    ax.title.set_text(
        ' w='+('50' if w2v_param == 0 else '200')
        +' ty='+a_type
        +' th='+str(count_threshold)
        +' m='+clustering_model_type
        +' p1='+str(model_param[0])
        +' p2='+str("none" if len(model_param) == 1 else model_param[1])
    )
    #ax.axis([np.partition(xg, 0)[0],xg.max(), -0.2, 1.2])
    scnt += 1
