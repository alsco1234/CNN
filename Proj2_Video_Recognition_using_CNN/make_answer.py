import pandas as pd
import csv

import os
os.environ.setdefault('PATH', '')
import numpy as np

# label 저장된 txt 파일 불러오기 
label_file = 's3d/kinetics400_labels.txt'

with open(label_file) as f:
    label_list = f.readlines()

# label 400개가 들어있는 리스트
label_list = [label.rstrip('\n') for label in label_list]  

df = pd.read_csv('kinetics_softmax_S3D.csv')
df_path = df['dir']

file = open('kinetics_answer.csv', 'a')
writer = csv.writer(file)
answer = []

# 각 video의 answer list 생성
for path in df_path:
    answer.append(path)
    for label in label_list:
        if label in path:
            answer.append(1)
        else: answer.append(0)
    writer.writerow(answer)
    answer = [] # 다음 video를 위해 초기화 

file.close()