# I3D

Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset

Joa ̃o Carreira†, .. (2017)

<aside>
💡 Two_Stream Inflated 3D ConvNets(I3D) : RGB + Optical Flow 개별 네트워크 예측값 평균

</aside>

<aside>
💡 Inflating : 필터의 dimension 늘리고, weight 1/N, Inception Block : 여러 크기 conv 병렬

</aside>

<aside>
💡 **이미지처럼 비디오도 transfer learning(전이 학습, pre-trained된 모델을 학습)하면 이득이 있고, 특히 kinetics dataset 사용하면 도움이 된다**

</aside>

```
Paper : [https://arxiv.org/abs/1507.05717](https://arxiv.org/abs/1507.05717)
code : [https://github.com/deepmind/kinetics-i3d](https://github.com/deepmind/kinetics-i3d) 
참고 리뷰 1 : [https://22-22.tistory.com/70](https://22-22.tistory.com/70)
👍참고 리뷰 2 : [https://eremo2002.github.io/paperreview/I3D/](https://eremo2002.github.io/paperreview/I3D/)
참고 리뷰 3 : [https://junsk1016.github.io/deeplearning/i3D-%EB%A6%AC%EB%B7%B0/#introduction](https://junsk1016.github.io/deeplearning/i3D-%EB%A6%AC%EB%B7%B0/#introduction)
참고 리뷰 4 : [https://everyview.tistory.com/30](https://everyview.tistory.com/30) 
C3D->I3D 연구동향 : [https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=khm159&logNo=222027509486](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=khm159&logNo=222027509486)
한국어발표영상 : [https://www.youtube.com/watch?v=kr5i-Wtf5Go](https://www.youtube.com/watch?v=kr5i-Wtf5Go) 
```

---

## 0.   Abstract

- **새로운 data set, kinetics**
    - 소규모 (UCF-101, HMDB-51) 와 달리 400개의 행동, 클래스당 400개 이상 클립이 유튜브에서 수집됨
    - kinetics로 train 하면 소규모 data set에서 성능이 얼마나 좋아지나?
- **새로운 network, Two-Steram Inflated 3D ConvNet (I3D)**
    - *3D ConvNet의 시공간확장 + Imagenet 구조, 파라미터 이용*

## 1. Introduction

- ImageNet을 통해 큰 데이터셋으로 이미지 훈련하면 다른 구조에도 써먹는다 알게됨. 이미지 뿐 아니라 비디오도 같지 않을까? 레이블 400개에 클립 400개 이상의 유튜브 영상, 두배 큰 데이터셋 kinetics 이용하자
- 여러 net kinetics로 학습한 후 소규모 dataset에서 미세 조정
    
    ⇒ 사전 훈련에 의한 성능 향상은 구조 따라 다르다!
    
    ⇒ 사전 훈련으로 최고의 성능 뽑아내는 모델, Two-steam Inflated 3D ConvNets(I3D) : 
    
    **이미지 분류, 3D Conv로 시공간** +  **Inception-v1에 기반한 I3D 모델로 kinetics에 대한 사전 훈련 최고의 효과** 
    

---

## 2. Action Classification Architectures

![스크린샷 2022-07-25 오후 1.45.02.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.45.02.png)

- 비디오 분석 구조 구분
    - Conv, layers operator : 2D(이미지 기반) or 3D(비디오 기반)?
    - network input : RGB video or precomputed [optical flow](https://gaussian37.github.io/vision-concept-optical_flow/)(객체움직임)?
    - 2D convnet이라면, 정보가 [LSTM](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr) 사용 or feature aggreation over time?
- 이미지 분류 네트워크 (Inception, ResNet)도 시공간으로 확장(Infate)가능
    
    ⇒ 사전 학습된 ImageNet 구조를 기반으로 Inception-v1 batch normalization 하면 구분 잘 될 것!
    

### 1) The Old 1 : ConvNet + LSTM

![스크린샷 2022-07-25 오전 10.12.02.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_10.12.02.png)

- LSTM
    - **프레임별 독립적 특징 추출은 시간적 구조를 무시**함. ⇒ 반복 계층을 추가해 상태와 시간 순서, 장거리 종속성을 가져야 함.
- LSTM + ConvNet
    - Inception-V1의 마지막 avarege pooling layer 뒤에 batch norm을 가진 LSTM배치
    - 맨위에 fc layer 추가
    - 프레임 5/25, 마지막 결과 프레임만, [cross-entropy-losses](https://wandb.ai/wandb_fc/korean/reports/---VmlldzoxNDI4NDUx)(발견된 확률분포 - 예측한 확률분포)로 테스트
    - 높은 수준 변화 o, 미세한 **낮은 수준 움직임** x
    - [Backpropagation-through-time](http://solarisailab.com/archives/1451)(에러합)위해 여러 프레임의 네트워크 펼쳐야 ⇒**train 비용 많**이듦

### 2) The Old 2 : 3D ConvNets

![스크린샷 2022-07-25 오전 10.11.08.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_10.11.08.png)

- C3D : 시공간 필터가 있는 표준 convnet과 비슷한 비디오 모델링
    - **시공간 데이터의 계층적 표현을 직접 만듦**
    - **conv 차원 추가로 parameter 늘어 train 어려움**
    - **사전 훈련 이점 x ⇒ 얕은 구조로 처음부터 train해야**
- kinetics 평가 위해 **원형 모델에서 변형**
    
    ![IMG_38044EE941D1-1.jpeg](I3D%205b2e59b3e37240a596a78fd630b970a6/IMG_38044EE941D1-1.jpeg)
    
    - 모든 conv, fc 뒤에 batch norm
    - 첫 pooling에서 kernel depth 1→2 (시간 신경씀)
    
    ⇒ 표준 K40 GPU 사용, 배치당 15개 비디오로 교육 ㄱㄴ
    

### 3) The Old 3 : Two-Stream Networks

![스크린샷 2022-07-25 오전 10.29.14.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_10.29.14.png)

- **Two-Stream(이전)**
    - ImageNet으로 Pre-train된 ConvNet 두개 복제본 통과 ⇒  1개의 **RGB frame** **+** 10개의 겉으로 계산된 **opticalflow**(객체움직임) frame의 스택으로부터 예측 평균 ⇒ 비디오의 짧은 시간 스냅샷 모델링
    - flow stream은 flow frame보다 input chanel이 2배(수평, 수직) 더 많은 adaptive input conv layer ⇒ 여러 스냅샷 샘플링, 예측 평균화 ⇒ **높은 성능, train, test 효율**
- **3D-Fused Two-Stream(최근)**

![스크린샷 2022-07-25 오전 11.20.58.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_11.20.58.png)

- 맨뒤 conv layer 뒤에서 시공간 + flow stream 합침 ⇒ 테스트 시간 줄임, HMDB 성능높
- **평가 위해 변형**
    - Inception-V1 사용, 10프레임 간격 연속 RGB frame(optical flow랑 비슷하게)
    - 3*3*3conv layer(512개출력) ⇒ 3*3 mac pooling layer ⇒ fc layer ⇒ Inception-V1 맨뒤 average pooling layer(5*7*7)
    - Gaussian noise로 초기화
    - 모두 [end-to-end학습(종단간학습, 입력출력한번에)](https://velog.io/@jeewoo1025/What-is-end-to-end-deep-learning)

### 4) The New : Two-Steam Inflated 3D ConvNets

C3D가 ImageNet 2D ConvNet 설계 + 선택적 train된 Parameter로부터 어떤 이득? 

C3D에 RGB + optical flow(직접 시간 패턴 학습 대신)로 성능 향상 

- **Inflating 2D ConvNets into 3D**
    - spatio-temporal model 반복 ⇒ 모델 자체를 2D에서 3D로
    - 2D 구조에서 모든 **filter, polling kernal 확장(inflate)** : 시간 차원추가
    - NxN filter ⇒ N*N*N filter
- **Bootstrapping 3D filters from 2D filters**
    - 2D⇒3D 구조 그대로 filter 확장 ⇒ pre-trained ImageNet model 파라미터 사용
    - 3D filter은 2D filter에 시간 축 추가해 확장한 것(boring) ⇒**2D filter의 weight값을 3D filter의 시간 축 따라 복사하고 이를 1/N로 rescaling**
- **Pacing receptive field growth in space, time and net-work depth**
    - video는 인접한 frame끼리 유사 ⇒ 모든 frame이 독립적, 유의미는 x
    - pooling과 conv layer에서 temporal stride 얼마로 줄 것인지 중요
        - 시간정보가 공간 비해 더 빠르게 감지되면 : 초기 특징 감지 x
        - 시간정보가 공간 비해 더 느리게 감지되면 : 장면 역할 감지 x
    - 실험적으로 **첫 두개의 max pooling에서는 시간 x가 효과적**임을 발견
        
        ![스크린샷 2022-07-25 오후 1.26.37.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.26.37.png)
        
    - **즉, Inception은 여러 개의 convolution layer 병렬적으로 사용해 연산량 조절**
        
        ![스크린샷 2022-07-25 오후 1.17.41.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.17.41.png)
        
    - [Inception-V1(GoogleNet)](https://oi.readthedocs.io/en/latest/computer_vision/cnn/googlenet.html) : Sparse network and dense matrix
    - 그래서 이렇게 max-pool 첫 두개 stride 1,2,2(커널크기는 1*3*3), 최종 average pooling layer이 2*7*7커널이 되게끔
- **Two 3D Streams**
    
    ![스크린샷 2022-07-25 오후 1.25.38.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.25.38.png)
    
    - I3D는 **RGB를 입력으로 받는 네트워크**와, **optical flow(객체움직임)을 입력으로 받는 네트워크** 두 개로 이루어짐 : two stream network
    - 두 네트워크 **개별 학습**, 두 **예측결과 average**

### 5) Implementation Details

- 실험에서
    - C3D 제외, 모두 Imagenet으로 pre-trained Inceptopn-v1 기본 네트워크로
    - 모든 구조에서 마지막 conv layer제외하고 batch norm과 relu함
    - 0.9 [SGC(Stochastic Gradient Descent, 확률경사하강법, 무작위 예의 예측 경사 계산)](https://everyday-deeplearning.tistory.com/entry/SGD-Stochastic-Gradient-Descent-%ED%99%95%EB%A5%A0%EC%A0%81-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95)
    - 32개 GPU 병렬화
    - kinetics 데이터 세트, tensorflow
    - 공간적 무작위 자르기 (256*256⇒ 224*224), 비디오 루프, 좌우 뒤집기, 좌우반전, 측광
    - optical flow : TV-L1(Total Variation L1, 노이즈 제거해서 객체 행동)

---

## 3. The Kinetics Human Action Video Dataset

- 활동, 사건보다 행동에 초점
- 세분화된 동작은 시간적 추론, 대상 강조 필요
- 400개 클래스, 각 클래스 마다 400개 이상의 clip, 각 클래스마다 총 240000개의 trian video. 10초, 모두 다듬어짐
- test set은 각 클래스별 100개 clip

---

## 4. Experimental Comparison of Architecture

![스크린샷 2022-07-25 오후 1.46.28.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.46.28.png)

- **I3D가 가장 잘 수행된다. RGB + optical flow 같이 쓸때 성능 가장 좋다**
- 모두 UCF-101보다 kinetics가 성능 낮다
    - HMDB-51 : train data 부족, 의도적으로 어려움
- 구조 순위 dataset 마다 일관적
    - kinetics와 달린 UCF, HMDB에서는 RGB 보다 optical flow가 더 낫다 : kinetics에는 카메라 motion이 다양해 motion stream 학습 어렵기 때문

![스크린샷 2022-07-25 오후 1.53.38.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.53.38.png)

- **imagenet으로 per-trained한 모델이 더 좋다**

---

## 5. Experimental Evaluation of Features

![스크린샷 2022-07-25 오후 1.58.40.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.58.40.png)

- pretrained 사용o / 사용x
- origin / fixed / full-ft
    - origin : UCF 혹은 HMDB에서 훈련됨
    - fixed : kinetics feature인데, 마지막 layer만 UCF나 HMDB에서 훈련됨
    - full-ft : UCF나 HMDB 미세조정후 end-to-end 학습한 kinetics로 훈련됨
- 모든 데이터, 특히 I3D와 C3D가 kinetics pre-trained(←) 이득얻음.
    - I3D는 high temporal resolution 사용하기 때문에 특히 굿. fps25인 비디오에서 64frame 샘플링하기 때문에 fine-grained temporal action feature(프로세스 나눠서 시간적 동작 정보) 잘 추출할 수 있음. 이와 달리 frame sampling이 적은 경우 kinetics dataset으로 pre-training시켰을 때 효과가 크지 않음
    - two-stream은 pre-training 효과 낫굿
        - flow stream이 이미 있어서 overfitting 안돼 정확도 굿
- **kinetics로 pre-training하는 것이 imagenet으로 pre-training하는 것 보다 낫다**

### 1) Comparision with the State-of-the-Art

![스크린샷 2022-07-25 오후 2.20.58.png](I3D%205b2e59b3e37240a596a78fd630b970a6/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2022-07-25_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_2.20.58.png)

- Two-Stream I3D가 이겼따
- kinetics pre-trained I3D가 sports 1m pre-trained C3D 이겼따
    - **I3D구조, kinetics dataset이 더 쎄다**

---

## 6. Discussion

- 이미지넷처럼 비디오 도메인에서도 전이학습(transfer learning, pre-trained된 모델학습)하면 이득 있을까? ⇒ kinetics 이용하면 이득
    - **kinetics dataset으로 pre-training하고 다른 데이터셋 fine-tuning 하면 굿**
- 그러나 kinetics로 pre-training하는것이 video segmentation, video object detection, optical flow computation 등 다른 비디오 task에서도 이득인지는 몰?루
- I3D라는 효과적인 모델
    - comprehensive exploration 수행 x
    - action tube나 attention mechanism같이 human actor에 focus 할 수 있는 테크닉 적용 x
    - space와 time 간의 관계 분석하고 중요한 정보만 감지하는 기술로 추가 연구 필요

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict

class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

# *** InceptionModule ***
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)

class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits
        

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)
```