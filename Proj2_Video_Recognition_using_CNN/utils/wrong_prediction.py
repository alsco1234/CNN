import os
import csv

f = open('./s3d_wrong_prediction.csv')
path_reader = csv.reader(f)
classes = dict()
total = 0

for line in path_reader:
    now_class = line[0].split('/')[-2]
    if classes.get(now_class) == None:
        classes[now_class] = 1
        total += 1
    else:
        classes[now_class] += 1
        total += 1

for key, val in classes.items():
    print(key, val)

print("## TOTAL IS", total)
f.close()