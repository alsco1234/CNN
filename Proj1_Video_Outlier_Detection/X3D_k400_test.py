import torch
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os
from urllib import request  # requires python3
from torch.fx import symbolic_trace
from os import path
from pathlib import Path
import json
import cv2
import csv
import natsort

# kinetics test json파일 열기
with open('../../../projects/vode/data/kinetics-dataset/k400/annotations/test.json', 'r') as f:
    json_data = json.load(f)
key_list = list(json_data.keys())

# 사용할 x3d모델 지정(x3d_s, x3d_xs, x3d_m)
model_name = 'x3d_m'
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

import json
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

# Set to GPU or CPU
#device = "cuda:0"
device = "cpu"
model = model.eval()
model = model.to(device)

json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try: urllib.URLopener().retrieve(json_url, json_filename)
except: urllib.request.urlretrieve(json_url, json_filename)


with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")
    # print(kinetics_id_to_classname[v], "v:", v)


mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 30
model_transform_params  = {
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}

# 모델의 base transform parameter불러오기
transform_params = model_transform_params[model_name]

# 원하는 대로 데이터를 핸들링 하기 위해, Compose기능 사용(rescale, randomCrop, ...)
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCropVideo(
                crop_size=(transform_params["crop_size"], transform_params["crop_size"])
            )
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second

# test video path설정
dir_name = os.listdir('../../../projects/vode/data/kinetics400/test')
dir_list = list(dir_name)
dir_list = natsort.natsorted(dir_list)

# 사용할 변수 선언
Total_video_num = 0
Correct_count = 0

def get_duration(filename):
    video = cv2.VideoCapture(filename)
    return video.get(cv2.CAP_PROP_POS_MSEC)

def prediction(video_path):
    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration
    videoPath = []
    # Initialize an EncodedVideo helper class and load the video
    #video = EncodedVideo.from_path(video_path)
    print("Video path: ", video_path)
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = inputs.to(device)

    # Pass the input clip through the model
    preds = model(inputs[None, ...])

    # Get the predicted classes
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    y = preds.detach().cpu().numpy()
    videoPath.append(video_path)
    df = pd.DataFrame(y)
    df['PATH'] = videoPath
    col1 = df.columns[-1:].to_list()
    col2 = df.columns[:-1].to_list()
    new_col = col1 + col2
    df = df[new_col]
    
    if not os.path.exists('kinetics_softmax_x3d_m.csv'):
        df.to_csv('kinetics_softmax_x3d_m.csv', index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv('kinetics_softmax_x3d_m.csv', index=False, mode='a', encoding='utf-8-sig', header=False)
    
    pred_classes = preds.topk(k=1).indices[0]

    # 예측한 Class 저장
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]

    # 예측한 Class 출력
    # print("Top 1 predicted labels: %s" % ", ".join(pred_class_names))
    
    for key in key_list:
        if key in video_path:
            label = json_data[key]['annotations']['label']

    # 실제 정답과 비교
    if pred_class_names == [label]:
        return 1
    else:
        return 0

for dir in dir_list:
    dir_path = '../../../projects/vode/data/kinetics400/test/' + dir
    file_path = []
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            file_path.append(os.path.join(root, file) )
    print("Workspace:",dir_path)
    for i in range (0, len(file_path)):
        video_path = file_path[i]
        vidcap =  cv2.VideoCapture(video_path)

        scene_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print('video frame: ', scene_length)
        if scene_length > 0:
            Total_video_num += 1
            Correct_count += prediction(video_path) # 예측과 정답이 맞으면 +1

# 최종 결과
print("\n*Result*\n")
print('Total video numbers: ', Total_video_num)
print('Correct video numbers: ', Correct_count)
print('Accuracy: ', Correct_count/Total_video_num)  # 맞은 비디오 수 / 사용한 전체 비디오 수