import numpy as np
import pandas as pd
import sys,os
from IPython import embed
from tqdm import tqdm as monitor

version = 'v3-'

problem_number = sys.argv[1]
mode_number = '3'
model_name = sys.argv[2] #"old-50-10-A" #"50-10-A-SVM-0.1"
features_path = './data/processed/csj/'+problem_number+'/clustering-'+model_name
dic_path = './data/external/hybrid.htkdic'
updated_dic_path = './data/processed/csj/'+problem_number+'/model-'+mode_number+'-clustering-'+version+model_name

print("1/3 read")
features = pd.read_csv(features_path, header=None)

print("2/3 calc point")
# Add update-point to features
features = features.iloc[0:4200]
f_upper = features.iloc[20-1][1]
f_lower = features.iloc[4200-1][1]
f_range = f_upper - f_lower
#features['point'] = [max(1.0, 0.5 + min(2.5, (feature - f_lower) / f_range * 2.5 )) for feature in features[1]] # 1.0~3.0
#features['point'] = [max(2.0, 2.0 + min(1.0, (feature - f_lower) / f_range * 1.0 )) for feature in features[1]]  # 2.0~3.0
features['point'] = [2.0 for feature in features[1]]  # 2.0
# Make dictionary
features_dic = {str(feature[0]).replace(' ',''):str(feature['point']) for i, feature in features.iterrows()}

print('3/3 update model')
cnt = 0
with open(dic_path) as fin:
    with open(updated_dic_path, 'w') as fout:
        for line in monitor(fin):
            if cnt < 3:
                fout.write(line)
            elif cnt < 64274:
                data = line.split('\t')
                word = data[0].split('+')[0]
                if word in features_dic:
                    if len(data) == 3:
                        fout.write(data[0]+'\t@'+features_dic[word]+'\t'+data[1][1:-1]+'\t'+'\t'.join(data[1:]))
                    elif len(data) == 5:
                        # @が元からついていた場合
                        prev_point = float(data[1].replace('@',''))
                        point = float(features_dic[word])
                        new_point = point * (10**prev_point)
                        fout.write(data[0]+'\t@'+str(new_point)+'\t'+'\t'.join(data[2:]))
                    else:
                        raise('?')
                else:
                    fout.write(line)
            else:
                data = line.split('\t')
                word = data[0]
                if word in features_dic:
                    fout.write(data[0]+'\t@'+features_dic[word]+'\t'+'\t'.join(data[1:]))
                else:
                    fout.write(line)
            cnt += 1

