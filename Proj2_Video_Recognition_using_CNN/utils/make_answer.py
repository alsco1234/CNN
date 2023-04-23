import pandas as pd
import csv

import os
os.environ.setdefault('PATH', '')
import numpy as np

# label 저장된 txt 파일 불러오기 
label_file = 'K400_labels.txt'

with open(label_file) as f:
    label_list = f.readlines()

# label 400개가 들어있는 리스트
label_list = [label.rstrip('\n') for label in label_list]  

df = pd.read_csv('/Users/kimminchae/Desktop/CNN/k400_top1.csv')
df_path = df['PATH']

file = open('r21d_answer.csv', 'a')
writer = csv.writer(file)
answer = []

# 각 video의 answer list 생성
for path in df_path:
    answer.append(path)
    for label in label_list:
        if ('/'+label+'/') in path:
            answer.append(1)
        else: answer.append(0)
    writer.writerow(answer)
    answer = [] # 다음 video를 위해 초기화 

file.close()