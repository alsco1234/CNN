import json
import os
from os import path
import shutil
from pathlib import Path

with open('../../../projects/vode/data/kinetics-dataset/k400/annotations/validate.json', 'r') as f:
    json_data = json.load(f)

original = '../../../projects/vode/data/kinetics-dataset/k400/val/'
target = '../../../projects/vode/data/kinetics400/val/'

# json파일 key(영상이름) 리스트 생성
key_list = list(json_data.keys())


file_names = os.listdir('../../../projects/vode/data/kinetics-dataset/k400/val')
file_list = list(file_names)


# 각 비디오 클래스별로 분류
for file in file_list:
  for key in key_list:
    if key in file:
      dir_path = Path("../../../projects/vode/data/kinetics400/val/" + json_data[key]['annotations']['label'])
      if dir_path.is_dir() == True:
        shutil.move(original+file, target +json_data[key]['annotations']['label'] + '/' + file)
      else:
        os.makedirs('../../../projects/vode/data/kinetics400/val/' + json_data[key]['annotations']['label'])
        shutil.move(original+file, target +json_data[key]['annotations']['label'] + '/' + file)