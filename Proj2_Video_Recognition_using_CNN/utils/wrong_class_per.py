"""
result example

abseiling  :  0.656
air drumming  :  0.54
answering questions  :  0.235
applauding  :  0.284
...

(맞은영상/전체영상) = 맞은 퍼센트

fisrt f = predicton
path, 0.001, 0.002 ... (401 col)

se
"""

import os
import csv

predic_class = dict()
answer_class = dict()
pred_total = 0
ans_total = 0

f = open('./s3d_wrong_prediction.csv')
path_reader = csv.reader(f)

for line in path_reader:
    now_class = line[0].split('/')[-2]
    if predic_class.get(now_class) == None:
        predic_class[now_class] = 1
        pred_total += 1
    else:
        predic_class[now_class] += 1
        pred_total += 1
f.close()
# for key, val in predic_class.items():
#     print(key, val)

# print("## TOTAL IS", total)

f = open("./i3d+r21d+x3d_path.txt")
txtlines = f.readlines()

for line in txtlines:
    now_class = line.split('/')[-2]
    if answer_class.get(now_class) == None:
        answer_class[now_class] = 1
        ans_total += 1
    else:
        answer_class[now_class] += 1
        ans_total += 1
f.close()

percent_class = dict()
for key, val in answer_class.items():
    if predic_class.get(key) == None:
        percent_class[key] = 1.0
    else:
        percent_class[key] = round((val - predic_class[key]) / val,3)
        print(key, " : ", percent_class[key])