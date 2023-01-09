# C3D

Learning Spatiotemporal Features with 3D Convolutional Networks

Du Tran, .. (2015)

> CNN은 `Convolution`과 `Pooling`을 반복적으로 사용하면서 불변하는 특징을 찾고, 그 특징을 입력데이터로 Fully-connected 신경망에 보내 `Classification`을 수행합니다.
> 

```
Paper : https://arxiv.org/pdf/1412.0767.pdf
참고 리뷰 1 : https://22-22.tistory.com/74
참고 리뷰 2 : https://m.blog.naver.com/khm159/221871148618
CNN 참고 : http://taewan.kim/post/cnn/
영상 feature 비교 : https://darkpgmr.tistory.com/116
레이어별 설명 : https://blog.naver.com/PostView.nhn?blogId=intelliz&logNo=221709190464
```

---

## 0.   Abstract

- Large scale, supervised 비디오 데이터셋에 대해 3차원 convolution network를 이용한 간단하고 효율적인 모델
- 이 3D Convnet은
    
    1) 2D ConvNet보다 시공간적 특징에 적합하다
    
    2) 모든 계층에 3*3*3 Conv kernel이 있는 구조는 3D ConvNet에 적합하다
    
    3) 간단한 선형 분류기를 가진 학습된 기능인 C3D는 4개의 다른 벤치마크에서 최고이며 다른 2개의 벤치마크에서도 훌륭하다
    
- feature은 간단하며, 속도는 빠르다. 개념적으로 간단하며, 훈련하고 사용하기 쉽다.

---

## 1.   Introduction

- 대규모 비디오 행동 인식, 비정상 사건 감지 등의 분석 기법이 필요하다.
- video descripter의 조건은 4가지이다 :
    
    1) 차별적이고 일반적. (ex. 음식, 스포츠, 영화..)
    
    2) 확장 가능한 작업을 처리할 수 있도록 작아야
    
    3) 효율적 계산 가능해야
    
    4) 선형 분류기와 같은 간단한 모델에서도 작동할 수 있게 간단해야
    
- 이미지 기반 딥 러닝은 행동 모델링이 부족해 비디오에 적합하지 않다. (섹션 4,5,6). 따라서 **간단한 선형 분류기를 가지고 시공간 기능을 학습**할 것을 제안한다. 이전에 제안된 3D ConvNet과 달리 이는 비디오의 객체, 장면, 동작 정보를 캡슐화하여 다양하게 작업할 수 있다. 이것은 일반적이고, 작고, 단순하고, 효율적이다. 요약하자면 이것의 장점은 :
    
    1) **C3D는 외형과 동작을 동시에 모델링 (two-stream)**
    
    2) 경험적으로 모든 계층이 잘 작동하는 3*3*3 Convloution
    
    3) 간단한 선형 모델은 4개의 작업들과 6개의 기준점에서 훌륭함
    
    ![스크린샷 2022-07-18 오후 4.54.03.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.54.03.png)
    
    성능이 이전 모델을 능가하며, optical flow, improved Dense Trajectory 기능 등을 추가로 사용할 때 성능이 좋다
    
    4) 계산하기에 작고 효율적
    

---

## 2.   Related Work

- **STIP** : 시공간적 관심점 검출 (비디오에 대한 관심 지점 감지)
- **SIFT-3D** : 이미지 크기, 회전해도 불변하는 특징 추출
- **HOG-3D** : 시공간에서 3D 모양의 기울기 방향 히스토그램으로 행동 인식
- **Harris corner detectors** : 모서리 인식해 이미지 특징 추출
- **Cuboids features for behavior recognition** : 직육면체로 행동 인식
- **action bank** : 샘플링된 개별 행동들의 집합
- **Improved Dense Trajectories(IDT)** : 동작 인식을 위한 세밀한 궤적들
- **fisher vector** : 데이터들을 하나의 직선(1차원 공간)에 projection(투사)시킨 후 그 투사된 된 data들이 구분되는지 판별
- **hand-crafted feature** : 선과 모서리 등 이미지가 급변하는 시점에서 검출
- **iDT descriptor** : 모서리인식 3D확대x → 특징점 샘플링 광학 흐름 측정 →대규모x
- **ConvNets** : 인간 자세 추정, 이미지 기능 학습으로 확대
- **stacked ISA** : ISA 네트워크 이용, 이미지 패치로부터 학습 → 입력 차원 작아야
- **Restricted Boltzmann Machine(RBM)** : 확률밀도함수 생성 → 이미지 생성
- **two stream networks : 장면과 물체의 공간적 부분 + 관찰자(카메라)와 물체의 움직임의 시간적 부분으로 구조를 두 부분으로 나눔**
- **3D ConvNets** : 사람탐지와 머리tracking으로 인간 피사체 분할, 동작 분류 위해 이 피사체는 3D ConvNet의 입력이 됨. but 이 논문에서는 **전체 비디오 프레임을 입력으로 사용**하고 **전처리에 의존하지 않음** → 대규모o.

이 논문에서는 **네트워크의 모든 계층에 시간 정보를 전파**하는 **3D-conv와 3D-pooling**을 수행한다. **공간 및 시간 정보를 점진적으로 풀링**하고 더 깊은 네트워크를 구축하는 것이 좋다.

---

## 3.   Learning Features with 3D ConvNets

C3D 기본 작동, 다양한 구조 경험적 분석, 대규모에서 훈련하여 학습하는 방법?

### 3.1 3D Convolution and pooling

[convolution](https://www.notion.so/convolution-0ed6545cce2548e5836e0b3ffa1dae4f)

[pooling](https://www.notion.so/pooling-cad7bdd0b65048a5af70b3f0fc2ba3d6)

![스크린샷 2022-07-19 오전 10.02.03.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_10.02.03.png)

2D ConvNets는 conv 직후에 입력 신호의 시간 정보를 손실하지만 3D conv는 보존한다. pooling도 마찬가지다. 복수의 frame을 가져도 2D Conv를 거치면 반드시 시간 정보를 손실한다. 이 논문에서는 중간 규모의 UCF101을 실험하여 경험적으로 최고의 구조를 찾아낸다. 2D보다 3*3의 Conv kernel이 가장 좋은 결과를 보였기 때문에, **3D Conv kernel을 3*3으로 고정하고 시간적 깊이만 변화시킨다.**

- **Notations**
    - 비디오 크기 = c * l * h * w
        
        (c = 채널 개수, l = 프레임 개수, h, w = 프레임 가로 세로)
        
    - 3D conv and pooling kernel size = d * k * k
        
        (d = 커널 깊이, k = 커널 사이즈) ←여기서 k는 3으로 고정, d만변화
        
- **Common network settings**
    
    UCF101에 대해
    
    - 입력 비디오 크기 = 3 * 16 * 128 * 171, 예외로 3 * 16 * 112 * 112도 넣음
    - 네트워크 = 5개의 conv layer + 5개의 pooling layer + 2개의 softmax loss layer
    - 5개의 conv layer의 필터 수 = 64, 128, 256, 256, 256
    - 커널 깊이 = d로 표현, 변화시킬 대상
    - **conv layer**의 공간시간 모두 stride(필터순회간격)=1이므로 **입력 출력 크기 변화x**
    - **pooling layer**
        
        첫 번째 1*2*2, 시간 일단 무시하고 16개의 클립 길이 만족해야
        
        나머지 2*2*2, stride=1의 max polling = 입력에서 **출력 8배 감소**함
        
        ⇒ 2048개의 출력.
        
    - epoch(전체 데이터 ?번학습) 16 거침
- **Varying network architectures**
    
    어떻게 시간 정보를 수집할것인가? 설정 유지하며 d만 바꿔보자.
    
    - 구조 1 ) conv layer의 시간적 커널 깊이가 동일하다 (depth-d)
        
        d = {1-1-1-1-1}, {3-3-3-3-3}, {5-..}, {7-..} (depth-1은 2D와 같다)
        
    - 구조 2 ) conv layer의 시간적 커널 깊이가 변화한다
        
        d = 증가형{3-3-5-5-7}, 감소형{7-5-5-3-3}
        
        모든 네트워크가 마지막 pooling layer의 출력 크기는 같다 (2048로)
        
        ⇒ 연결된 레이어에 대한 매개 변수 수는 동일하다
        
        ⇒ 매개 변수의 수는 conv 레이어에서만 다르다 (d가 다르니까). 완전히 연결된 레이어에서는 미미하다. 네트워크의 학습 용량이 비교 가능하고 매개 변수 수의 차이가 구조 검색에 영향을 주지 않아야 한다.
        

### 3.2 Exploring kernel temporal depth

그 결과

![스크린샷 2022-07-19 오전 11.11.46.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.11.46.png)

depth-1(2D ConvNet)이 성능이 가장 나쁘다. 3*3 kernel에 depth가 3(3 * 3 * 3)kernel이 가장 좋다. (경험적으로 depth-3이 최상이다)

### 3.3 Spatiotemporal feature learning

- **Network architecture**
    
    ![스크린샷 2022-07-19 오전 11.16.55.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.16.55.png)
    
    위의 결과대로 구조를 만들어서 대규모 데이터 세트를 사용한다. **8개의 3D ConvNet레이어(3*3*3, stride=1*1*1), 5개의 pooling layer(1*2*2*, stride=2*2*2(처음만 stride1*2*2)), 2개의 완전 연결된(pool없는)fc 레이어(4096개 출력), softmax 출력 레이어**이다. 
    
- **Dataset**
    
    대량의 sports-1M 사용
    
- **Training**
    - 비디오 크기 = 3 * 5 * 128 * 171, 공간 및 시간 위해 3 * 16 * 112 * 112로 무작위 자름, 반은 수평으로 뒤집음
    - SGD(Stochastic Gradient Descent, 확률적 기울기 선형)으로 30개의 작은 batch(데이터셋 하나에 들어간 데이터 수)크기로 분류해서 수행됨. 13epoch에서 최적화됨
- **Sports-1M classification result**
    
    ![스크린샷 2022-07-19 오전 11.38.52.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.38.52.png)
    
    C3D는 상위 5개 비디오 레벨 정확도에서 Slow Fusion을 능가한다. 120frame은 너무 길어서 C3D와 직접 비교할 수 없다. 처음부터 train된 C3D는 84의, 사전 훈련된 모델의 미세 조정된 C3D는 85.2의 정확도를 가진다. 
    
- **C3D video descriptor**
    
    train된 C3D는 다른 비디오 분석 기능 추출기로 사용할 수 있다. 기능을 추출하기 위해 비디오는 두 개의 연속된 클립 사이에 8프레임 오버랩이 있는 16프레임의 긴 클립으로 분할하고, C3D로 넣어 soft-max(fc)6 활성화를 추출한다. 이 활성화는 4096-dim 비디오 기술자를 형성하기 위해 평균화된 다음 L2-정규화를 따른다.
    
- **What does C3D learn?**
    
    deconvolusion을 해보자. C3D는 처음 몇 프레임의 모습에 초점을 맞추고 후속 프레임에서 두드러진 움직임을 추적한다. 
    
    ![스크린샷 2022-07-19 오전 11.47.14.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.47.14.png)
    
    처음 몇 프레임에서는 모양을, 나머지는 움직임에만 주의를 기울인다. C3D는 **동작과 외관 모두 선택적으로 참여**한다는 점에서 2D ConvNet과 다르다.
    

---

## 4.   Action recognition

- **Dataset**
    
    다시 UCF101(중간크기데이터). 세 가지 분할 설정 사용
    
- **Classification model**
    
    C3D기능 → multi-class linear SVM (Support Vector Machine, 이진 선형 분류 모델)
    
    C3D 구분자 세가지
    
    1) I380k에서 훈련된 C3D
    
    2) Sports-1M에서 훈련된 C3D
    
    3) I380k에서 훈련되고 Sports-1M에서 미세 조정된 C3D
    
    여러개의 network를 설정할 때, L2-normalized(직선거리) C3D 구분자들을 연결한다
    
- **Baselines**
    
    C3D기능을 몇 기준으로 비교해보자. Imagenet(IDT)이다. IDT는 여러 영상 특징에 대해 각 채널이 500개의 사이즈를 갖는다. L1-norm(절댓값합) 이용하여 히스토그램 정규화하고 이에 대한 25k feature vector 만든다. 이 프레임 기능을 평균화하여 비디오 구분자 만든다. 비교를 위해 multi-class linear SVM 사용.
    
- **Results**
    
    ![스크린샷 2022-07-19 오후 1.19.10.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.19.10.png)
    
    상단 : linear SVM만 간단히 이용, 중간 : RGB 프레임만 입력, 하단: multiple feature 조합
    
    - C3D(1 net) < C3D(3 net) : dimension 4096 → 12288
    - C3D(1 net) + Imagenet = LRCN(연속적으로 나타나는 이미지 순차적 처리) (0.6향상) ⇒ 이미 C3D가 외형과 동작 모두 잘 캡쳐하므로 외형 심층 기능 Imagenet 결합해도 이득 없다
    - C3D(3 net) + IDT = 90.4 ⇒ C3D와 IDT는 낮은 수준 gradient의 히스토그램, C3D는 높은 수준의 추상/의미 정보 캡쳐하므로 상호보완성 뛰어나다
    - C3D(3 net)는 RGB와 optical flow를 둘 다 이용할때 RNN보다도 낫다
    - 단순하다는 장점도 있다
    - 그러나 C3D는 two-stream networks, IDT-based method, long-term modeling등과 결합해야 한다.
- **C3D is compact**
    
    ![스크린샷 2022-07-19 오후 1.41.36.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.41.36.png)
    
    PCA(차원감소) 한 후 linear-SVM 이용, UCF101에 투영된 기능의 분류 정확도. 낮은 차원에서 더 성능이 좋다 ⇒ **낮은 저장비용과 빠른 검색이 중요한 대규모 데이터셋에 적합하다**
    
    ![스크린샷 2022-07-19 오후 1.51.40.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.51.40.png)
    
    Feature Embedding(시각화). 같은 색 = 같은 동작. 데이터 무작위 클립 따와서 fc6기능 추출한 후 t-SNE(차원 2차원으로 축소)해서 그림. ⇒ **C3D는 일반화 잘됨**
    

---

## 5.   Action Similarity Labeling

- **Dataset**
    
    ASLAN 데이터 세트 (두 개의 비디오 보고, 얘네 같은 행동인지 아닌지 예측). 처음 본 행동 보고 유사성을 검사
    
- **Feature**
    
    비디오를 8프레임의 중첩이 있는 16프레임 클립으로 나눔. prob, fc7, fc6, pool5와 같은 C3D기능을 추출. 각 기능 유형에 대해 클립 기능을 개별적으로 평균화한 다음 L2-norm으로 계산
    
- **Classification model**
    
    4가지 유형의 기능의 12개의 다른 거리를 계산하여, 12*4의 기능 벡터를 얻고, 비교할 수 있게 정규화한다음, linear-SVM이 다르거나 같게 분류하도록 훈련한다. 이 외에도 C3D와 동일한 baseline과 비교한다
    
- **Results**
    
    ![스크린샷 2022-07-19 오후 2.11.45.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.11.45.png)
    
    정확도도, AUC(분류모델의성능)도 C3D가 상당하다
    
    ![스크린샷 2022-07-19 오후 2.11.58.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.11.58.png)
    
    현재 최첨단 기술을 능가한다
    

---

## 6.   Scene and Object Recognition

- **Dataset**
    
    1인칭 시점의 영상에서 장면을 인식해보자
    
- **Classification model**
    - 같은 특징 추출, linear SVM, leave-one-out(LOOCV, 1개의 데이터만 test set, 나머지 전부 학습)평가 프로토콜
    - 기능 추출 위해 16프레임 길이로 비디오 클립 찍고, 16프레임 창 슬라이드하고, 실측 사실 레이블을 가장 자주 발생하는 레이블로(8프레임 이상만)
    - linear SVM으로 훈련하고 테스트, 객체 인식 정확도 보고
    - Imagenet 기능 사용하여 기준선과 비교
- **Results**
    
    ![스크린샷 2022-07-19 오후 2.39.02.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.39.02.png)
    
    당시 최신 기술 [9]를 능가한다. [9]는 서로 다른 복잡한 기능 인코딩(FV, LLC, 동적 pooling)을 사용하고, C3D는 클립 기능의 단순 균화를 가진 선형 SVM만 사용하는 방법인데도 말이다. **C3D는 일반적인 모양과 동작 정보를 비디오로 감지할 수 있다.**
    

---

## 7.   Runtime Analysis

![스크린샷 2022-07-19 오후 2.53.10.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.53.10.png)

^UCF101에 대한 runtime분석. C3D는 Real-Time보다 **빠르다**

---

## 8.   Conclusion

3D ConvNet(3CN)을 이용해 비디오의 시공간 문제를 학습했다.

최적의 시간 커널 길이를 찾기 위해 체계적인 연구를 수행했다

C3D가 외모와 동작 정보를 동시에 모델링 할 수 있으며, 다양한 비디오 분석 작업에서 2D ConvNet 기능을 능가할 수 있음을 보여주었다.

linear classifier가 있는 C3D기능이 다른 비디오 분석 기준점에서 당시 최상의 방법을 능가하거나 비슷함을 증명했다

결론적으로, 제안된 **C3D는 효율적이고, 작으며, 간단하고, 빠르다**

---

## Appendix A : Effects of Input Resolution

입력 해상도를 바꾸면 어떻게 될까? 3의 실험에서, kernel depth가 아닌 프레임 크기(h, w)을 변경해보자

- net-64 (64*64크기), net-128, net-256으로 실험하며, 모두 depth-3이다
    
    ![스크린샷 2022-07-19 오후 3.01.29.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.01.29.png)
    
- 마지막 pooling layer에서 추력 크기가 다르므로 매개 변수가 크게 차이난다. 이에 따라 훈련하는데 시간이 차이난다.
    
    ![스크린샷 2022-07-19 오후 3.01.57.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.01.57.png)
    
- 경험상 net-128이 가장 적절했다. 훈련 시간, 정확도, 메모리 소비 사이의 균형을 맞춰야 한다. GPU 메모리 제한이 있다면 모델 병렬화를 해야 한다.

## Appendix B : Visualization of C3D Learned Features

C3D가 내부적으로 학습한 내용을 더 잘 이해하기 위해 제공함

- **Decovolutionss of C3D**
    
    UCF101에서 무작위로 클립을 선택한 후 미리 선택된 conv layer에서 동일한 feature map에 대해 강하게 보이는 클립을 그룹화하여 이 클립의 상단 활성화를 이미지 공간에 다시 투영. 
    
    ![스크린샷 2022-07-19 오후 3.07.51.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.07.51.png)
    
    처음 두줄은 움직이는 모서리와 모양을 감지하지만 셋째줄은 샷 변화, 모서리 방향 변화 및 색상 변화를 감지한다.
    
    ![스크린샷 2022-07-19 오후 3.08.44.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.08.44.png)
    
    각 그룹은 conv3b에서 학습한 기능 맵으로, 맨 위 feature map은 움직이는 모서리와 움직이는 질감을 감지한다. 중간은 움직이는 신체를, 맨 아래는 객체 궤적 및 원형 객체를 탐지한다.
    
    ![스크린샷 2022-07-19 오후 3.11.34.png](C3D%208e19e68b731749eda59b4b491fa4f1f5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-07-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.11.34.png)
    
    물체의 움직임을 감지하는 C3D conv5b의 deconvolution train feature map. 두번째의 마지막 클립에서는 움직이는 머리를 감지하고, 마지막 클립에서는 움직이는 헤어컬러를 감지한다.
    
- 결론적으로 conv2a에서 C3D는 이동 모서리, 짧은 변화, 모서리의 변화 또는 색상 변화화 같은 낮은 수준의 움직임 패턴을 학습한다. 하지만 conv3b의 상위 계층에서 C3D는 모서리, 텍스처, 신체 부위 및 궤적의 더 큰 이동 패턴을 학습한다. 마지막으로, 가장 깊은 컨볼루션 레이어인 conv5b에서 C3D는 움직이는 원형 물체, 자전거와 같은 움직임 등 더 복잡한 움직임 패턴을 학습한다.

---

```python
import torch.nn as nn

class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        """
            8개의 3D ConvNet 레이어 (3*3*3, stride=1*1*1)
                각 특징 파악해 feature map 만듬
            5개의 pooling layer(2*2*2, stride=2*2*2)....처음만 1*2*2
                중요한 feature만 남김
            2개의 fc 레이어
                2차원을 1차원으로 평탄화
                이전 레이어의 모든 노드가 연결되었기때문에 fully connected
            softmax 출력 레이어
        """
        
        # 커널 사이즈 = d * k * k, 여기서 k=3으로 고정, d만변화

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)) 
        # conv layer 1: 필터수 64, padding으로 크기 작아지지않게
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        # 첫번째 pooling에서는 1,2,2로 일단 시간 무시, polling과 stride는 같아야 모든 원소 처리. 이때 크기 감소함

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # conv layer 2 : 필터수 128
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) # max pooling으로 입력에서 출력 감소

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # conv layer 3 : 필터수 256
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # conv layer 2 : 필터수 256
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # conv layer 2 : 필터수 256
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        # 논문대로 여기서 4096개 출력
        self.fc8 = nn.Linear(4096, 487)
        # 이미지 출력 위해 fc

        self.dropout = nn.Dropout(p=0.5) # overfitting 방지

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1) # 0~1의 확률로

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192) # reshape, -1로 적절한 행 생성
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs
```