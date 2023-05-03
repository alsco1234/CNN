import csv
import torch
import torchvision
from timesformer.models.vit import TimeSformer
import numpy as np
import torch
import torchvision.io as io

# 1. 모델 불러오기
model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='/Users/kimminchae/Desktop/CNN/TimeSformer_divST_8x32_224_K400.pyth')

# 2. txt에서 Path 읽어서
fr = open("K400.txt", 'r')
while True:
    video_path = fr.readline()
    video_path = video_path[:-1] # \n제거
    if not video_path: break

    # 3. 비디오 파일 읽기
    video, audio, info = io.read_video(video_path)
    del audio, info

    # 4. 비디오 텐서 정규화하기
    video2 = video.permute(0, 3, 1, 2).float() / 255.0
    del video

    # 5. n번째 segment
    video3 = video2[0:8, :, :, :]
    del video2

    # 6. 차원 맞추기
    video4 = video3.unsqueeze(dim=0) # 1, 8, 3, 720, 360
    del video3
    video5 = video4.transpose(1, 2) # (프레임 수, 채널 수, 높이, 너비) 1, 3, 8, 720, 406
    del video4

    # 7. softmax 뽑기
    pred = model(video5,) # (1, 400)
    del video5
    
    # 8. csv에 추가하기 
    numpy_array = pred.detach().numpy()
    np.savetxt("result.csv", numpy_array, delimiter=",")
    del numpy_array

fr.close()