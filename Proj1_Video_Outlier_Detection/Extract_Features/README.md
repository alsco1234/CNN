# 이상점 검출 기반 비디오 축약기법 개발   

**k400 pretrained x3d로 feature 뽑아 outlier detection하는 과정**   

1. **원본 데이터 받기**   
    1. 원본 데이터 화질이 너무 좋을 경우 압축 (**[https://www.veed.io/ko-KR/tools/video-compressor](https://www.veed.io/ko-KR/tools/video-compressor)**)   
    2. 프레임 전부 추출해 저장 (/home/alsco1234/x3d-dashcam/video_2_images.py)   
    3. 추출한 프레임을 원하는 만큼 건너뛰어 저장 (/home/alsco1234/x3d-dashcam/frames_pick.py)   
    4. 프레임을 다시 영상으로 저장 (/home/alsco1234/x3d-dashcam/images_2_video.py)   
2. **다듬어진 데이터로 feature 추출**   
    1. .npy파일로 추출 (/home/alsco1234/x3d-dashcam/Extract-Features-X3D/x3d_extract_main3_reframe.py )   
        1. 이때 CCD의 경우 3개 붙어있는 feature segment 단위로 쪼개야함. crash의 경우 crash-normal과 crash-crash를 분리함. (/home/s21900395/video_features/split_feature.py)   
    2. .csv파일로 다시 저장 (?)     
        1. 추가로 원래 있던 normal csv에 붙이던가 하는거 가능   
3. **feature 분석 + outlier detection**   
    1. feature map 그려서 대충 성능 확인 (https://colab.research.google.com/drive/   1htfBvqpklQ3Z3NWS3ORZaBSFH3NfBSt2?authuser=3)   
    2. IForest 돌리기 ([https://colab.research.google.com/drive/1r7oL9ploRXvDI-mJAmb8z09-PMCougea?authuser=3](https://colab.research.google.com/drive/1r7oL9ploRXvDI-mJAmb8z09-PMCougea?authuser=3))   
    3. OC-SVM 돌리기   

(https://colab.research.google.com/drive/1PI1qgsvjDExTIre6tzGJYio4Tk7ZScKD?authuser=3)   

[데이터 저장된 위치](https://www.notion.so/442c94e2c2014b7195d94ef37198819a)   