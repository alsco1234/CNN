import csv
import torch
import torchvision
from timesformer.models.vit import TimeSformer
import numpy as np
import torch
import torchvision.io as io
from torchinfo import summary

# 1. 모델 불러오기
model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='/Users/kimminchae/Desktop/CNN/TimeSformer_divST_8x32_224_K400.pyth')

batch_size = 16
summaary(model, input_size=(16, 3, 224, 224))