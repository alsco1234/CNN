# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import numpy as np
import natsort
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# 파일을 읽기전 csv파일에 Column이 없다면 local에서 따로 Column을 지정해주어야 함 (PATH, 0~399)
kinetics_predict = pd.read_csv('kinetics_s3d_top10_test.csv')  # 예측 닶
kinetics_answer = pd.read_csv('kinetics_answer.csv') # 실제 답

Acc = 0
Precision = 0
Recall = 0
f1 = 0

#--------------------column 돌아가면서 성능 계산---------------------------
for i in range(1, 401):
  Predict = kinetics_predict.iloc[i]
  Answer = kinetics_answer.iloc[i]

  pred_list = Predict.values.tolist()
  ans_list = Answer.values.tolist()

  #pred_list = sum(pred_list, [])
  #ans_list = sum(ans_list, [])

  correct = 0
  for i in range(0, len(pred_list)):
    if pred_list[i] == 1 and ans_list[i] == 1:
      correct += 1
  Acc += correct/len(pred_list)
  Precision += round(precision_score(ans_list,pred_list,average='micro'),3)
  Recall += round(recall_score(ans_list,pred_list,average='micro'),3)
  f1 += round(f1_score(ans_list,pred_list,average='micro'),3)
  print("**Current Result**")
  print("Accuracy: ", Acc)
  print("Precision: ", Precision/400)
  print("Recall: ", Recall/400)
  print("f1: ", f1/400, "\n")

print("=============S3D Total Result=============")
print("Accuracy: ", Acc)
print("Precision: ", Precision/400)
print("Recall: ", Recall/400)
print("f1: ", f1/400)