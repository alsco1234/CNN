import csv
import torch
import torchvision
from timesformer.models.vit import TimeSformer
import numpy as np
import torch
import torchvision.io as io
import math
import time
import gc
import torch

start = time.time()

# 1. 모델 불러오기
device = torch.device("cuda")
model = TimeSformer(img_size=224, num_classes=768, num_frames=8, attention_type='divided_space_time',  pretrained_model='/home/alsco1234/TimeSformer/TimeSformer_divST_8x32_224_K400.pyth')
model.to(device)
model.eval() # 불필요 레이어 무시
torch.no_grad() # 메모리 절약

# 2. txt에서 Path 읽어서
fr = open("/home/s21900395/i3d+r21d+x3d_path.txt", 'r')
j = 0

while True:
    j = j + 1
    video_path = fr.readline()
    video_path = video_path[:-1] # \n제거
    if not video_path: break

    # 3. 비디오 파일 읽기
    video, audio, info = io.read_video(video_path)
    video = video.cuda()
    del audio, info

    # 4. 비디오 텐서 정규화하기
    video2 = video.permute(0, 3, 1, 2).float() / 255.0
    del video

    frames = video2.shape[0]
    segs = int(frames/8)

    # 5. n번째 segment
    for i in range(1,segs+1):
        print(f'this video {i} / ',segs, end='\r')
        video3 = video2[8*(i-1):8*i, :, :, :]
        #del video2

        # 6. 차원 맞추기
        video4 = video3.unsqueeze(dim=0) # 1, 8, 3, 720, 360
        del video3
        video5 = video4.transpose(1, 2) # (프레임 수, 채널 수, 높이, 너비) 1, 3, 8, 720, 406
        del video4

        # 7. softmax 뽑기
        with torch.no_grad():  
            pred = model(video5,) # (1, 400)
        del video5
        
        # 8. csv에 추가하기 
        numpy_array = pred.tolist() #pred.detach().numpy() 보다 메모리 덜 소모
        del pred
        numpy_array = np.insert(numpy_array, 0, i-1) # segment 번호 (i)
        numpy_array2 = np.concatenate(([video_path], numpy_array))# 영상 주소
        del numpy_array

        with open('testoneesult.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(numpy_array2)
        del numpy_array2
    
    print('all video ', j, ' / 3000 ')
    break # test one video

fr.close()

end = time.time()
print(f"{end - start:.5f} sec")