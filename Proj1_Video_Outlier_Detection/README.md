# 3D CNN 모델을 이용한 동영상 내 행동인식

## 재실험 : segment voting

이전 실험의 문제 : 각 Model마다 한 segment에 들어가는 frame 수가 다른데 (64, 32, 16 ..) 영상의 첫 segment로만 판단하여 판단되는 영상의 길이가 다름
재실험 방법 : 영상의 모든 segment를 각각 softmax를 뽑아, 이중에서 voting을 하자

---

### 1. 사용한 영상

- /home/s21900395/i3d+r21d+x3d_path.txt
- 총 34342개 (* 영상의 segment 개수)

### 2. segment 뽑는 방법

- fps는 원본 그대로 둠
- 한 영상의 총 프레임이 200이고, s3d가 64frame을 가지니 64 + 64 + 64 + 8에서, 64가 되지않는 8은 버림

### 3. voting 방법

- soft voting

---

### s3d_new_sofrmax

- softmax의 csv 파일 위치 : /home/alsco1234/s3d2/s3d_0217ver/s3d_new_sofrmax.csv
- 위치 : /home/alsco1234/s3d2/s3d_0217ver
- 실행명령어 : python main.py feature_type = s3d
- 가상환경 : x
- Screen name : s3d_2017ver
- 실행시간
    - 시작 : 2월 17일 10시 32분
    - 총 소요시간 8시간 08분 13.511882초
        
        ![스크린샷 2023-02-28 오전 10.28.06.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/60b69da1-83ee-45cb-9a4c-9b21659439ad/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-02-28_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_10.28.06.png)
        
    - 끝시간 : 2월 17일 18시 40분
- 특이점
    - r21d와 다르게 main에서 실행시간 측정
    - r21d와 같이 npy저장
    - Extract_s3d클래스의 extract 함수에서 각 세그먼트별 output을 두번 뽑는 과정이 있어 생략함 (line 74-77)
    - ~~Extract_s3d클래스의 extract 함수에서 fps변경하고 저장해서 불러오는 과정 생략함 (line 56-58) : yml에서 keep_tmp_files = false하면됨. 해결함~~
    - ‘The value is empty for s3d’ (issue 91, 미해결) 때문에 npy파일이 제대로 저장이 안됨. csv파일은 잘 되는데.. 일단 뽑아봐야 알것같긴함. ([https://github.com/v-iashin/video_features/issues/91](https://github.com/v-iashin/video_features/issues/91))
- 결과
    
    ![스크린샷 2023-02-17 오전 11.08.07.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5e4a3ff3-3212-4fb7-acd9-3f0828400184/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-02-17_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11.08.07.png)
    
    (오름차순 정렬은 임의로 한거)

### R(2+1)D_new_sofrmax

### x3d_new_sofrmax