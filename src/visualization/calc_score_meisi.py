# -*- coding: utf-8 -*-

import lxml.html
from lxml import etree
import sys, os, re
import requests

def read(path):
    f = open(path)
    data = f.read()
    f.close()
    return data

print("1/2 Get diff")
answer_file_path = sys.argv[1]
recog_file_path = sys.argv[2]
dist_path = sys.argv[3]
answer = read(answer_file_path)
recog = read(recog_file_path)
data = {
    'sequenceA':answer,
    'sequenceB':recog
}
response = requests.post('https://difff.jp/',data)
print("Html:response:"+str(response.status_code))
with open(dist_path+"/diff.html", 'w') as f:
  f.write(response.text)

print("2/2 Calc diff")
html = read(dist_path+'/diff.html')
html = html.replace('&nbsp;','')
dom = lxml.html.fromstring(html)
line_list = dom.xpath('//*[@id="result"]/table/tr')

cnt=0

def check(str):
    if str == "、" or str == "。" or str == "" or str == "\n":
        return False
    else:
        return True

correct_cnt = 0
error_cnt = 0

for line in line_list:
    text = line.xpath('td[1]')[0]
    word_list = str(lxml.html.tostring(text)).split(' ')
    if word_list != ["b'<td></td>\\n\\t'"] and 'font' not in word_list[0]: #skip
        print("*********************************************************")
        isEmMode = False
        for word in word_list:
            word = word.replace('&#12290;','').replace('&#12289;','')
            word = word.replace('<td>','').replace('b\'','')
            word = word.replace('<em></em>',"") #空白の削除
            if word == '':
                continue
            #if '</td>' in word:
            #    continue
            if '<em>' in word:
                isEmMode = True
            if word[-4:] != '<em>' and isEmMode:
                print("em :"+word)
                error_cnt += 1
            else:
                print("not :"+word)
                correct_cnt += 1
            if '</em>' in word:
                isEmMode = False
print(error_cnt)
print(correct_cnt)
print(correct_cnt*100/(error_cnt+correct_cnt))
