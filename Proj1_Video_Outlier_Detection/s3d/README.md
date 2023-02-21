<h1>TODO</h1>
<h2>### s3d_new_sofrmax 진행과정

- 위치 : /home/alsco1234/s3d2/s3d_0217ver
- 실행명령어 : python main.py feature_type = s3d
- 가상환경 : x
- Screen name : s3d_2017ver
- 실행시간
    - 시작 : 2월 17일 10시 32분
    - 예상 : 약 10시간
    - 끝시간 : 약 2월 17일 20시 32분
- 특이점
    - r21d와 다르게 main에서 실행시간 측정
    - r21d와 같이 npy저장
    - Extract_s3d클래스의 extract 함수에서 각 세그먼트별 output을 두번 뽑는 과정이 있어 생략함 (line 74-77)
    - ~~Extract_s3d클래스의 extract 함수에서 fps변경하고 저장해서 불러오는 과정 생략함 (line 56-58) : yml에서 keep_tmp_files = false하면됨. 해결함~~
    - ‘The value is empty for s3d’ (issue 91, 미해결) 때문에 npy파일이 제대로 저장이 안됨. csv파일은 잘 되는데.. 일단 뽑아봐야 알것같긴함.
  </h2>






문제정의 
(Problem definition)
블랙박스 영상 분석 비용 축소를 목적으로 블랙박스 영상에 video outlier detection모델을 사용하는 자동화 기술을 개발을 목적으로 한다.

i) Input: CCD dataset의 영상 데이터와 교통상황 감시카메라 혹은 직접 수집한 블랙박스 데이터를 사용할 모델의 학습에 알맞은 형태로 전처리 후, input데이터로 사용한다.

ii) function: 영상 분류 3D CNN모델 중 최고의 성능을 내는 모델을 선택하여 pre-trained모델을 찾고, 교통 법규를 위반한 차량사고 영상 데이터에 대해 fine-tuning을 진행한 뒤, feature extraction과정을 수행하여 input으로 들어온 사고 영상에 대한 특징을 추출한다. 추출된 영상의 특징은 영상당 하나의 1차원 array형태로 outlier detection 모델의 input으로 들어가게 되고, outlier detection모델에서는 이러한 특징들은 분석하여 실제로 사고가 발생하였거나, 교통 위반이 감지된 지점을 찾게 된다.

iii) output: 3D-CNN모델로 부터 영상 내 특징이 추출되고, 추출된 특징이 outlier detection 모델을 거치게 되면 최종 output이 나오게 된다. output은 각 영상의 특징을 분석하여 계산된 이상치를 기반으로 실제로 사고 장면이나, 교통 위반 장면이라고 의심되는 부분에 대한 영상내 지점을 반환하게 된다.



설계 목표
블랙박스 내(dash cam video)에서 불법 상황을 찾기 위하여 사람이 직접 긴 블랙박스 영상을 관찰하는 것은 많은 시간과 인력의 낭비를 초래한다는 문제점이 있다. 이에 대한 근거로, 최근 경찰청에서 공개한 데이터에 의하면, 2021년 교통법규 위반처리 건수는 1700만건 이상이며 공익 신고는 290만건으로 공익 신고를 도와주는 기술 개발의 필요성이 보이며, 이를 통해 교통법규 위반 상황이 줄어드는 효과를 기대해 볼 수 있다. 이러한 기능을 하기 위해 본 캡스톤 팀은 모델의 정확성을 최우선순위로 설정함으로써, 보다 정확한 교통 법규 위반 상황을 인식할 수 있도록 하는것에 목표를 두었다. 

해당
설계요소
목표설정
블랙박스 영상 분석 비용 축소를 목적으로 블랙박스 영상에 video outlier detection모델을 사용하는 자동화 기술을 개발을 목적으로 한다.
분석
시장조사기관 트렌드모니터 등에 따르면, 국내 차량의 블랙박스 설치율은 90%이며, 블랙박스 장착을 의무화 하려는 국가가 증가하는 추세인 것을 확인할 수 있다. 또한 블랙박스가 범죄 예방과 사고 수습 등에서 사용되면서, 꾸준한 성장세를 보이고 있는 가운데 아직까지 블랙박스 내(dash cam video)에서 불법 상황을 찾기 위하여 사람이 직접 긴 블랙박스 영상을 관찰하는 것은 많은 시간과 인력의 낭비를 초래한다는 문제점이 있다. 이에 대한 근거로, 최근 경찰청에서 공개한 데이터에 의하면, 2021년 교통법규 위반처리 건수는 1700만건 이상이며 공익 신고는 290만건으로 공익 신고를 도와주는 기술 개발의 필요성이 보이며, 이를 통해 교통법규 위반 상황이 줄어드는 효과를 기대해 볼 수 있다.
개념 및 상세설계
영상을 한 세그먼트 당 64개의 프레임으로 나누어 실험 결과를 통해 선정한 3D CNN 모델에 입력 값을 넣어 특징을 추출한다. 추출한 특징은 이미지를 처리하는 VAE-based Deep SVDD와 DASVDD(Deep Autoencoding SVDD)를 벤치마킹하여 영상 처리에 적합하도록 재설계한 VAE-SVDD 모델의 입력 값으로 넣으면 outlier 지점을 파악하여 사고의 시점을 찾아낼 수 있다.
구현 및 제작
i) Input: CCD dataset의 영상 데이터와 교통상황 감시카메라 혹은 직접 수집한 블랙박스 데이터를 사용할 모델의 학습에 알맞은 형태로 전처리 후, input데이터로 사용한다.

ii) function: 영상 분류 3D CNN모델 중 최고의 성능을 내는 모델을 선택하여 pre-trained모델을 찾고, 교통 법규를 위반한 차량사고 영상 데이터에 대해 fine-tuning을 진행한 뒤, feature extraction과정을 수행하여 input으로 들어온 사고 영상에 대한 특징을 추출한다. 추출된 영상의 특징은 영상당 하나의 1차원 array형태로 outlier detection 모델의 input으로 들어가게 되고, outlier detection모델에서는 이러한 특징들은 분석하여 실제로 사고가 발생하였거나, 교통 위반이 감지된 지점을 찾게 된다.

iii) output: 3D-CNN모델로 부터 영상 내 특징이 추출되고, 추출된 특징이 outlier detection 모델을 거치게 되면 최종 output이 나오게 된다. output은 각 영상의 특징을 분석하여 계산된 이상치를 기반으로 실제로 사고 장면이나, 교통 위반 장면이라고 의심되는 부분에 대한 영상내 지점을 반환하게 된다.


시험 및 평가
모델의 성능을 평가하는 지표로 AUROC를 선택하였다.




과목명
공학프로젝트기획
연도 및 학기
2022-2학기
설계주제
이상점 검출 기반 영상 축약 기법 개발 (Video Outlier Detection and Extraction)
팀원
(이름, 학번)
강현묵(22000017), 고영광(21700026), 김영민(21800121), 김민채(22000080), 박서휘(22000263), 신소은(21900395)
지도교수,
산업체자문위원
홍참길 교수님, ㅇㅇㅇ
문제정의 
(Problem definition)
블랙박스 영상 분석 비용 축소를 목적으로 블랙박스 영상에 video outlier detection모델을 사용하는 자동화 기술을 개발을 목적으로 한다.

i) Input: CCD dataset의 영상 데이터와 교통상황 감시카메라 혹은 직접 수집한 블랙박스 데이터를 사용할 모델의 학습에 알맞은 형태로 전처리 후, input데이터로 사용한다.

ii) function: 영상 분류 3D CNN모델 중 최고의 성능을 내는 모델을 선택하여 pre-trained모델을 찾고, 교통 법규를 위반한 차량사고 영상 데이터에 대해 fine-tuning을 진행한 뒤, feature extraction과정을 수행하여 input으로 들어온 사고 영상에 대한 특징을 추출한다. 추출된 영상의 특징은 영상당 하나의 1차원 array형태로 outlier detection 모델의 input으로 들어가게 되고, outlier detection모델에서는 이러한 특징들은 분석하여 실제로 사고가 발생하였거나, 교통 위반이 감지된 지점을 찾게 된다.

iii) output: 3D-CNN모델로 부터 영상 내 특징이 추출되고, 추출된 특징이 outlier detection 모델을 거치게 되면 최종 output이 나오게 된다. output은 각 영상의 특징을 분석하여 계산된 이상치를 기반으로 실제로 사고 장면이나, 교통 위반 장면이라고 의심되는 부분에 대한 영상내 지점을 반환하게 된다.


설계 목표
블랙박스 내(dash cam video)에서 불법 상황을 찾기 위하여 사람이 직접 긴 블랙박스 영상을 관찰하는 것은 많은 시간과 인력의 낭비를 초래한다는 문제점이 있다. 이에 대한 근거로, 최근 경찰청에서 공개한 데이터에 의하면, 2021년 교통법규 위반처리 건수는 1700만건 이상이며 공익 신고는 290만건으로 공익 신고를 도와주는 기술 개발의 필요성이 보이며, 이를 통해 교통법규 위반 상황이 줄어드는 효과를 기대해 볼 수 있다. 이러한 기능을 하기 위해 본 캡스톤 팀은 모델의 정확성을 최우선순위로 설정함으로써, 보다 정확한 교통 법규 위반 상황을 인식할 수 있도록 하는것에 목표를 두었다. 
해당
설계요소
목표설정
블랙박스 영상 분석 비용 축소를 목적으로 블랙박스 영상에 video outlier detection모델을 사용하는 자동화 기술을 개발을 목적으로 한다.
분석
시장조사기관 트렌드모니터 등에 따르면, 국내 차량의 블랙박스 설치율은 90%이며, 블랙박스 장착을 의무화 하려는 국가가 증가하는 추세인 것을 확인할 수 있다. 또한 블랙박스가 범죄 예방과 사고 수습 등에서 사용되면서, 꾸준한 성장세를 보이고 있는 가운데 아직까지 블랙박스 내(dash cam video)에서 불법 상황을 찾기 위하여 사람이 직접 긴 블랙박스 영상을 관찰하는 것은 많은 시간과 인력의 낭비를 초래한다는 문제점이 있다. 이에 대한 근거로, 최근 경찰청에서 공개한 데이터에 의하면, 2021년 교통법규 위반처리 건수는 1700만건 이상이며 공익 신고는 290만건으로 공익 신고를 도와주는 기술 개발의 필요성이 보이며, 이를 통해 교통법규 위반 상황이 줄어드는 효과를 기대해 볼 수 있다.
개념 및 상세설계
영상을 한 세그먼트 당 64개의 프레임으로 나누어 실험 결과를 통해 선정한 3D CNN 모델에 입력 값을 넣어 특징을 추출한다. 추출한 특징은 이미지를 처리하는 VAE-based Deep SVDD와 DASVDD(Deep Autoencoding SVDD)를 벤치마킹하여 영상 처리에 적합하도록 재설계한 VAE-SVDD 모델의 입력 값으로 넣으면 outlier 지점을 파악하여 사고의 시점을 찾아낼 수 있다.
구현 및 제작
i) Input: CCD dataset의 영상 데이터와 교통상황 감시카메라 혹은 직접 수집한 블랙박스 데이터를 사용할 모델의 학습에 알맞은 형태로 전처리 후, input데이터로 사용한다.

ii) function: 영상 분류 3D CNN모델 중 최고의 성능을 내는 모델을 선택하여 pre-trained모델을 찾고, 교통 법규를 위반한 차량사고 영상 데이터에 대해 fine-tuning을 진행한 뒤, feature extraction과정을 수행하여 input으로 들어온 사고 영상에 대한 특징을 추출한다. 추출된 영상의 특징은 영상당 하나의 1차원 array형태로 outlier detection 모델의 input으로 들어가게 되고, outlier detection모델에서는 이러한 특징들은 분석하여 실제로 사고가 발생하였거나, 교통 위반이 감지된 지점을 찾게 된다.

iii) output: 3D-CNN모델로 부터 영상 내 특징이 추출되고, 추출된 특징이 outlier detection 모델을 거치게 되면 최종 output이 나오게 된다. output은 각 영상의 특징을 분석하여 계산된 이상치를 기반으로 실제로 사고 장면이나, 교통 위반 장면이라고 의심되는 부분에 대한 영상내 지점을 반환하게 된다.


시험 및 평가
모델의 성능을 평가하는 지표로 AUROC를 선택하였다.
기타


제약조건
개발환경
OS: Debian GNU/Linux 10
CPU: AMD Ryzen 9 3900XT 12-Core Processor
RAM: 128G
GPU: GeForce RTX 3090
운영환경


제작비용 및 기간
약 6개월
사용성 및 심미성
사람이 직접 관찰하여 찾아내는 기존의 방식에서, 자동으로 이상치를 검출해줌으로써, 인력 절감과 시간적으로 효율적으로 사용할 수 있다.
사회 및 윤리성
블랙박스 내 교통 법규 위반 사항을 자동으로 검출할 수 있다면, 공익 신고 수도 현재 상황보다 늘어날 것으로 기대할 수 있기 때문에, 사회적으로 운전자들이 교통법규를 준수하는 분위기가 조성될 것이라 기대된다.
기타


예상 
산출물
// 고객에게 전달될 최종 결과물의 형태
// 캡스톤까지 성공적으로 마친 후 최종적으로 남겨야 하는 프로젝트 결과의 형태 (예: 실행화일, 소스화일, 설계도면, 사용자 매뉴얼, 유지보수 매뉴얼, 부품 리스트 등)
예상 산출물은 python으로 구현된 약 4개의 코드 파일(run.py(model.py + outlier.py), segmentation.py, model.py, outlier.py)이다. run.py 파일은 사용자가 최종적으로 실행하게 되는 실행파일로, 영상의 segment마다 feature를 추출하고, 추출한 feature에 대한 outlier score를 계산하여 최종 output형태로 반환한다. segmentation.py 파일은 검사 대상이 되는 영상을 여러개의 segment로 분할하는 작업을 수행하는 실행 파일이다. model.py 파일은 segment에 대한 feature를 추출하는 3D-CNN model를 불러와 실행시키는 파일이다. model.py 파일은 실행파일로 부터 추출한 feature를 통해 outlier score를 계산하는 작업을 수행하는 실행파일이다.


요약문



블랙박스 영상에 video outlier detection모델을 사용하는 자동화 기술을 개발한다. 전처리된 교통 상황 영상에 대해 feature extraction과정을 수행하여 outlier detection 모델의 input으로 이용하고, 특징들을 분석하여 사고가 발생하였거나, 교통 위반이 감지된 지점을 찾게 된다. 결과물로 계산된 이상치를 기반으로 교통 위반 장면이라고 의심되는 부분에 대한 영상내 지점을 반환힌디.


Summary of Capstone Design Project Proposal

Title
Video Outlier Detection and Extraction
Team members
Hyunmuk Kang, Youngkwang Ko, Minchae Kim, Youngmin Kim, Seohwee Park, Soeun Shin
Advisor
Charmgil Hong
Problem
definition
The goal is to develop an automation technology that uses a video outlier detection model for black box images to reduce the cost of analyzing black box images.

i) Input: After preprocessing the image data of the CCD dataset and the traffic monitoring camera or the directly collected black box data in a form suitable for learning the model to be used, it is used as input data.

ii) Function: Select the best performance model among 3D CNN models for image classification to find the pre-trained model, perform fine-tuning on vehicle accident image data that violate traffic regulations, and perform a feature extraction process to extract the characteristics of the accident image that came into input. The extracted features of the image enter the input of the outlier detection model in the form of one-dimensional array per image, and in the outlier detection model, these features are analyzed to find the point where an accident actually occurred or a traffic violation was detected.

iii) output: The feature in the image is extracted from the 3D-CNN model, and the final output is produced when the extracted feature goes through the outlier detection model. Based on the outliers calculated by analyzing the characteristics of each image, the output actually returns the point in the image for the accident scene or the suspected traffic violation scene.
Objectives
There is a problem that observing long black box images directly by a person in order to find illegal situations in a black box causes a lot of waste of time and manpower. As a basis for this, data released by the National Police Agency recently showed that more than 17 million traffic violations were handled in 2021 and 2.9 million public interest reports, indicating the need for technology development to help reduce traffic violations. To achieve this function, this Capstone team aimed to make the accuracy of the model a top priority, allowing them to recognize more accurate traffic violations.
Constraints
i) In the process of using 3D-CNN to extract features for accident images, it is difficult to find a pre-trained model with a dataset related to car accident images, which requires a process of fine-tuning the model directly.
ii) Video data that directly occurred in a traffic accident is relatively easy to obtain, but the number of video data that simply captures scenes that violate traffic regulations is small
Expected outcome
The expected output is approximately four code files implemented in python (model.py + outlier.py, segmentation.py, model.py, outlier.py). The run.py file is an executable file that the user finally executes, and the feature is extracted for each segment of the image, and the outlier score for the extracted feature is calculated and returned in the final output form. The segmentation.py file is an executable file that performs the operation of dividing an image to be inspected into several segments. The model.py file is a file that invokes and executes a 3D-CNN model that extracts features for segments. The model.py file is an executable file that performs the task of calculating the outlier score through the feature extracted from the executable.
Abstract of the proposed design project



We develop an automation technology that uses a video outlier detection model for black box images. A feature extraction process is performed on the preprocessed traffic situation image to be used as an input of the outlier detection model, and characteristics are analyzed to find a point where an accident occurred or a traffic violation was detected. Based on the outliers calculated as the result, the point in the image is returned for the suspected traffic violation scene.



목차


프로젝트 개요						// 1~2페이지 분량
1.1. 문제 정의 (Problem definition)
1.2. 목적과 목표 (Goals and Objectives)
1.3. 최종 산출물 (Deliverables)	
1.4. Key Project Stakeholders 
1.5. Project Requirements and Constraints 

프로젝트 배경 정보 						// 3~4페이지 분량
2.1.	기존의 유사 제품에 대한 조사
2.2.	기존의 연관 논문 및 특허 조사
2.3.	관련된 전공 지식 분야	
	
3.  개념 디자인과 전략						// 3~4페이지 분량 
3.1.	개념 설계
3.2.	설계의 타당성

4.  프로젝트 수행 계획						// 1~2페이지 분량
4.1. 수행해야 하는 작업
4.2 팀 협력 방안과 팀원별 역할

5. 필요한 자원							// 1~2페이지 분량
5.1.	구현 및 실험에 예상되는 소요 부품 리스트	
5.2.	프로젝트 수행에 필요한 예산 내역	

6. 첨부
6.1.	인용 자료 및 참고 문헌	
6.2.	실험 데이터, 수식 전개, 증명 등 세부 기술적인 사항들
6.3. 기타 첨부자료 




// 메모의 항목을 참고하여 10 페이지 이상 작성할 것., 작성 후 ‘파란색’ 주석 “// ~~”은 삭제할 것.


프로젝트 개요 
1.1. 문제 배경
시장조사기관 트렌드모니터 등에 따르면, 국내 차량의 블랙박스 설치율은 90%이며, 블랙박스 장착을 의무화 하려는 국가가 증가하는 추세인 것을 확인할 수 있다. 또한 블랙박스가 범죄 예방과 사고 수습 등에서 사용되면서, 꾸준한 성장세를 보이고 있는 가운데 아직까지 블랙박스 내(dash cam video)에서 불법 상황을 찾기 위하여 사람이 직접 긴 블랙박스 영상을 관찰하는 것은 많은 시간과 인력의 낭비를 초래한다는 문제점이 있다. 이에 대한 근거로, 최근 경찰청에서 공개한 데이터에 의하면, 2021년 교통법규 위반처리 건수는 1700만건 이상이며 공익 신고는 290만건으로 공익 신고를 도와주는 기술 개발의 필요성이 보이며, 이를 통해 교통법규 위반 상황이 줄어드는 효과를 기대해 볼 수 있다.



1.2. 문제 정의 (Problem definition)
블랙박스 영상 분석 비용 축소를 목적으로 블랙박스 영상에 video outlier detection모델을 사용하는 자동화 기술을 개발을 목적으로 한다.
i) Input: CCD dataset의 영상 데이터와 교통상황 감시카메라 혹은 직접 수집한 블랙박스 데이터를 사용할 모델의 학습에 알맞은 형태로 전처리 후, input데이터로 사용한다.

ii) function: 영상 분류 3D CNN모델 중 최고의 성능을 내는 모델을 선택하여 pre-trained모델을 찾고, 교통 법규를 위반한 차량사고 영상 데이터에 대해 fine-tuning을 진행한 뒤, feature extraction과정을 수행하여 input으로 들어온 사고 영상에 대한 특징을 추출한다. 추출된 영상의 특징은 영상당 하나의 1차원 array형태로 outlier detection 모델의 input으로 들어가게 되고, outlier detection모델에서는 이러한 특징들은 분석하여 실제로 사고가 발생하였거나, 교통 위반이 감지된 지점을 찾게 된다.

iii) output: 3D-CNN모델로 부터 영상 내 특징이 추출되고, 추출된 특징이 outlier detection 모델을 거치게 되면 최종 output이 나오게 된다. output은 각 영상의 특징을 분석하여 계산된 이상치를 기반으로 실제로 사고 장면이나, 교통 위반 장면이라고 의심되는 부분에 대한 영상내 지점을 반환하게 된다.









1.3. 최종 산출물 (Deliverables)
i) 실행파일: python으로 구현된 약 4개의 코드 파일(run.py(model.py + outlier.py), segmentation.py, model.py, outlier.py)
run.py: 사용자가 최종적으로 실행하게 되는 실행파일로, 영상의 segment마다 feature를 추출하고, 추출한 feature에 대한 outlier score를 계산하여 최종 output형태로 반환한다.
segmentation.py: 검사 대상이 되는 영상을 여러개의 segment로 분할하는 작업을 수행하는 실행 파일이다.
model.py: segment에 대한 feature를 추출하는 3D-CNN model를 불러와 실행시키는 파일이다.
model.py실행파일로 부터 추출한 feature를 통해 outlier score를 계산하는 작업을 수행하는 실행파일이다.
ii) 사용자 메뉴얼: 3D-CNN이나 Outlier detection model에 대한 사전 지식이 없이도, 프로그램을 사용할 수 있도록, 각 코드 파일의 용도에 대한 설명과 함께 실행 순서와 출력되는 결과물 형태에 대한 설명을 제공하는 메뉴얼을 제공한다.
iii) 설계 도면: 아래와 같은 방식으로 설계 도면을 제작하여 Get data, Feature extraction, Outlier detection, Evaluation, Service순으로 프로그램 설계 내용을 전달한다.


Get data: CCD dataset의 데이터와 교통상황 감시카메라 혹은 블랙박스 데이터를 직접 수집하여 학습에 알맞은 형태로 전처리 과정 수행
Feature extraction: 영상 분류 3D CNN모델 중 최고의 성능을 내는 모델을 선택하여 pre-trained모델을 찾고, 교통 법규를 위반한 차량사고 영상 데이터에 대해 fine-tuning을 진행하여 feature extractor를 개발
Outlier detection: 완성한 feature extractor에 여러 방법의 outlier detection을 적용하여 가장 성능이 좋은 방법으로 조합
Evaluation: Feature extractor모델과 outlier detection모델의 조합별 AUROC를 비교하고 class별 가장 좋은 성능의 모델 조합을 선정
Service: Desktop application이나 web service등 기업에서 사용할 수 있는 형태로 서비스화


1.4. 프로젝트 이해 관계자 
본 프로젝트의 잠재적 사용자는, 사고가 발생하였거나 교통 위반을 목격한 영상을 소유한 모든 사람들이 될 수 있지만, 특히 경찰 측에서 유사 상황 발생시 필요 인력을 줄이며 효율적인 작업을 가능하도록 하는 것에 목적을 두기 때문에 경찰을 포함한 교통 법규 관계자들을 주요 사용자로 정하였다.
1.5. Project Requirements and Constraints 
성능: 제안한 모델의 성능은 Top5기준 90%이상의 정확도를 보인다.

기능:
i) 동영상 데이터 segmentation: 사용자가 평가할 영상 데이터를 특정 프레임수로 나눠, 여러 segment로 분할
ii) feature extraction: input 영상으로 부터 만들어진 segment로 부터 특징 추출
iii) outlier scoring: 추출한 특징들로부터 outlier score를 매겨, 최종적으로 normal한 상황인지, unnormal한 상황인지 분류

제약 조건:
i) 사고 영상에 대한 특징을 추출하기 위해 3D-CNN을 사용하는 과정에서, 자동차 사고 영상과 관련된 데이터셋으로 사전 훈련된 모델을 찾기가 어려움, 이로인하여 모델을 직접 fine-tuning하는 과정이 필요함.
ii) 교통사고가 직접적으로 일어난 영상 데이터는 비교적 쉽게 구할 수 있지만, 단순히 교통 법규를 위반하는 장면을 포착한 영상 데이터들의 수가 적음


프로젝트 배경 정보
2.1.	기존의 유사 제품에 대한 조사
// 단순 내용 소개가 아니라 각기의 장단점을 분석해서 이 프로젝트의 방향설정과 연관지어야 함

2.2.	기존의 연관 논문 및 특허 조사
// 기존 논문 내용의 핵심 아이디어를 분석하고 본 과제와 연계해서 활용할 아이디어와 보완해서 발전시켜야 할 것으로 구분하여 기술할 것

논문

i. Unsupervised Traffic Accident Detection in First-Person Videos

  본 논문에서는 계속해서 움직이는 카메라(dash cam) 동영상에서 교통사고를 탐지하는 모델을 다룬다. 지금까지 동영상을 기반으로 이상 상황을 탐지하는 모델에 대한 연구는 많았지만 다음의 두 가지 이슈를 해결하지 못하였다. 
  첫 번째로, 기존의 방법들은 고정된 상태로 촬영하는 감시 카메라의 동영상을 사용하였기 때문에 자동차에 장착된 dash cam과 같이 움직이는 카메라에는 활용되지 못하였다. 두 번째로, 이상탐지를 할 때 이상 상황 별로 카테고리를 나누어 분류하는 방법을 사용하였기 때문에 사람이 직접 labeling 해놓은 trained dataset이 필요했다. 이는 supervised learning 형태이므로 시간과 비용이 많이 들고, 모델이 train되는 과정에서 보지 못한 데이터에 대해서는 이상 탐지가 되지 않는다는 단점이 있었다. 이러한 한계로 인해 도로 위의 사고를 예측하기 위해서는 새로운 모델이 필요했다.
  도로 위에서 일어날 수 있는 상황의 수는 거의 무한대에 가깝지만, 각 상황이 일어날 확률은 매우 적은 long-tailed distribution 형태를 가지고 있다. 흔하지 않은 상황들은 충분한 training data를 수집하기 어렵기 때문에 supervised learning 방법을 사용하여 이상 상황을 탐지하기에는 한계가 있다. 이러한 문제를 해결하기 위하여 논문에서는 정상 상황인지, 아니면 이상 상황에 대한 신호가 존재하는지의 이진 분류로 단순화한다. 만약 이상 상황이 발생한다면 모델이 예측한 위치에서 크게 벗어난 위치에 객체가 있을 것이라는 전제를 세우고, 모델에는  정상 주행 데이터만을 사용하여 학습시키기 때문에 데이터를 손쉽게 얻을 수 있을 뿐만 아니라 별다른 labeling 작업도 불필요하다.



  본 논문에서 제안하는 모델은 교통사고 혹은 비정상적인 상황이 발생할 경우 이를 감지하여 해당 객체를 빨간색 박스로 표시한다. 영상 프레임 내의 객체들에 대한 미래 위치를 예측하여 실제 위치와의 차이가 크면 비정상적인 상황이라고 판단한다. 움직이는 카메라에 적용시키기 위해 전체 프레임에 대한 예측이 아닌 객체의 행동 반경만 예측한다. 또한 dash cam이 장착된 ego-vehicle의 움직임에 대한 이상치도 탐지할 수 있다.
  논문에서 사용된 모델의 아이디어 중 본 과제와 연계하여 활용할 아이디어는 첫 번째로 bounding box prediction이다. Bounding box로 움직이는 객체들을 인식한 다음, 이상 상황이 발생하면 사고에 관련된 객체를  빨간색 bounding box로 표시하고, 해당 장면을 추출하는 방식으로 사용할 수 있다. 두 번째로는 missed object를 추적하는 방법이다. 본 논문에서는 동영상 프레임을 지나면서 순간 순간 탐지되는 모든 객체들을 리스트에 담아두어 특정 객체가 다른 객체에 의해 가려져 보이지 않게 되더라도 잊혀지지 않게 한다. 그 이유는 보이지 않는 순간에도 그 객체의 위치를 계속 추정할 수 있어야 사고를 놓치지 않을 수 있기 때문이다. 본 과제에서도 이 아이디어를 사용하여 보이지 않는 순간에도 차량의 예상 위치를 추적하여 이상 상황을 검출할 수 있다.
  보완하여 발전시킬 부분은 본 논문에서는 사고 장면만 검출했지만, 과제에서는 신호 위반이나 안전운전불이행 등 교통 법규 위반 상황도 검출하는 것을 목적으로 하고 있다. 이러한 상황들을 검출하기 위해서는 해당 객체의 미래 위치를 예상하는 것 뿐만 아니라 추가적인 정보가 더 필요하다.


ii.  Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning



  본 논문은 새로운 uncertainty-based 사고 예측 기법을 제안한다. 이는 움직이는 모든 물체를 후보로 두고 이들간의 관계로 미래의 사고를 예측하는 방법이다. 
  이 모델의 첫 번째 핵심 기법은 물체의 공간적/시간적 관계를 모두 고려하며 사고 예측을 위한 단서를 학습하는 것이다. 공간 관계는 시각 기억 + 공간 거리 + 외관 특징에 의해 학습되며, 시간 관계는 시간적 맥락에서 차량의 움직임이 어떻게 진화하고 사고로 끝나는지의 패턴에 의해 학습된다. 
  두 번째 핵심 기법은 RNN에 GCN을 결합한 cyclic process로 잠재적인 시공간 관계를 생성하는 것이다. 이는 3D CNN과는 다르게 시공간을 모두 학습할 수 있으며, BNN (Bayesian deep Neural Network)을 모델에 통합시켜 사고 점수를 예측한다.
  기존의 방법들은 deterministic(결정론적)한 방법들이지만, 본 모델은 움직이는 모든 물체를 사고 발생의 후보로 두고 predictive uncertainty를 추정하기 때문에 사고 발생을 예측하는 성능이 향상된다.
  본 논문에서 활용할 아이디어는 움직이는 물체들을 사고 발생 가능성이 있는 후보들로 두고 이들간의 관계로 미래에 일어날 사고를 예측하는 것이다. 또한 임계값을 두고 사고가 발생할 확률이 이 임계값을 처음 넘어서는 시간을 파악하여 사고가 발생하는 지점을 찾아낼 수 있다. 이 논문에서는 CCD(Car Crash Dataset)라는 새로운 데이터셋을 제안하는데, 이 데이터셋은 normal 동영상과 crash 동영상으로 구분되어 있으며 주석으로 각 영상의 환경 속성과 사고 이유 등의 정보를 가지고 있어 사용하기 편리하다. 이 데이터셋을 사용하여 본 과제에서 개발하는 모델을 학습시킬 수 있다.
  보완하여 발전시켜야 할 점은 본 논문에서는 RNN을 가지고 학습하기 때문에 객체의 위치가 이동하면 detection이 헷갈릴 수 있다는 점이다. 예를 들어 한 차량이 다른 차량에 가려져 사라졌다가 다시 나오는 경우 detection이 어렵게 된다. 움직이는 dash cam 동영상에는 객체가 주변의 객체들에 의해 가려졌다 나오는 상황이 많이 발생하는데, 객체가 화면에 보이지 않는 순간에도 움직임을 추적할 수 있어야 사고를 놓치지 않고 예측할 수 있다.


2.3.	관련된 전공 지식 분야

본 과제를 진행하기 위해서는 머신러닝(machine learning)에 대한 지식이 필요하다.
머신러닝이란 컴퓨터가 스스로 학습할 수 있도록 도와주는 알고리즘이나 기술을 개발하는 것을 말한다.

머신러닝은 일반적으로 아래와 같은 순서로 동작한다.

일정량 이상의 샘플 데이터를 입력
입력받은 데이터를 분석하여 일정한 패턴과 규칙을 파악
찾아낸 패턴과 규칙을 가지고 의사결정 및 예측 수행

머신러닝의 알고리즘은 다음과 같이 분류될 수 있다.

Unsupervised learning
Supervised learning
Semi-supervised learning

머신러닝에서 컴퓨터가 스스로 학습한다는 것은 입력받은 데이터를 분석하여 일정한 패턴이나 규칙을 찾아내는 과정을 의미한다. 이를 위해서는 사람이 인지하는 데이터를 컴퓨터가 인지할 수 있는 데이터로 변환해 주어야 하는데, 이 때 데이터별로 어떤 특징을 가지고 있는지 찾아내고 이것을 토대로 데이터를 벡터로 변환하는 작업을 특징 추출(feature extraction)이라고 한다.

https://www.ibm.com/cloud/learn/neural-networks

실험에서 사용된 3D CNN 모델들을 이해하기 위해서는 먼저 신경망(neural network)에 대한 이해가 필요하다.
Neural network란 인간의 뇌가 가진 뉴런의 연결 구조를 가리키며, 이러한 신경망을 본따 만든 구조를 인공 신경망(artificial neural network)이라고 부른다. 인공 신경망은 여러 뉴런이 서로 연결되어 있는 구조의 네트워크이며, input layer를 통해 학습하고자 하는 데이터를 입력받는다. 입력된 데이터들은 여러 단계의 hidden layer를 통과하며 처리되어 output layer를 통해 결과가 출력된다. 이러한 신경망을 3개 이상 중첩한 구조를 DNN(Deep Neural Network)라고 부르는데, 이를 활용한 머신러닝 학습을 딥러닝이라고 부른다.

그 다음으로, CNN(Convolutional Neural Network)은 일반 DNN에서 이미지와 동영상 등의 데이터를 처리할 때 발생하는 문제점들을 보완한 방법이다. DNN은 기본적으로 1차원 형태의 데이터를 사용하기 때문에 이미지를 사용하기 위해서는 이를 flatten시켜 한 줄의 데이터로 표현해야 한다. 이렇게 표현된 데이터는 공간적/지역적(spatial/topological) 정보가 손실된다. 이러한 문제점으로 부터 고안된 해결책이 CNN인데, CNN은 이미지를 그대로 받음으로써 공간적/지역적 정보를 유지한 채 feature들의 layer를 만든다. 또한 CNN의 중요한 특징은 이미지 전체보다는 부분을 보고, 이미지의 한 픽셀과 주변 픽셀들의 연관성을 살리는 것이다.


     
       https://www.ibm.com/cloud/learn/convolutional-neural-networks

CNN은 여러 개의 Convolution layer와 Pooling layer를 거쳐 이미지의 특징을 추출한다.
Convolution layer에서는 input image에 대해 kernel(또는 filter)를 기준으로 합성곱(convolution)을 하여 이미지의 feature map을 만든다. Pooling layer에서는 이미지의 크기를 적당히 줄이고 특정 feature를 강조하는 역할을 한다. 이 과정을 통해 특징을 추출하고, 최종적으로 fully connected layer를 통해 classification을 수행한다.

기존의 CNN 모델들은 2D 데이터를 사용하여 이미지에 담긴 공간적(spatial) 의존성을 추출하는 데에 목적을 두었다. 이러한 모델들은 이미지 데이터에 대해 효과적인 인식이 가능했지만 시간에 따라 연속적으로 변화하는 동영상 데이터의 인식에는 취약했다. 이러한 한계를 극복하기 위해 input 데이터와 kernel에 시간이라는 새로운 차원을 허용한 3D CNN이 제안되었고, 이로써 시공간적(spatio-temporal) 정보를 사용할 수 있게 되어 보다 정확한 동영상 분석이 가능하게 되었다. 3D CNN의 시작으로 볼 수 있는 C3D을 포함하여 본 과제의 실험에서 살펴본 P3D, R(2+1)D, S3D, X3D는 3D CNN의 대표적인 모델들이다.

	
3.  개념 디자인과 전략
3.1. 개념 설계
// 공프기에서 앞서 정의한 문제의 해결을 위해 노력한 내용 (인공지능 분야는 네트워크 모델 설계, 학습방법, 데이터 수집과정 등이 포함되어야 함. 웹/앱개발의 경우, UI설계, 메뉴 계층구조 설계, 데이터베이스 스키마 정의 등이 포함되어야 함. 임베디드의 경우, 회로설계도나 서비스 구현을 위한 수단이 제시되어야 함.

블랙박스 영상 분석 비용 축소를 목적으로 블랙박스 영상에 video outlier detection모델을 사용하는 자동화 기술을 개발하기 위해 다음과 같이 모델을 설계하고자 한다.




우선 영상을 한 세그먼트 당 64개의 프레임으로 나누어 실험 결과를 통해 선정한 3D CNN 모델에 입력 값을 넣어 특징을 추출한다. 추출한 특징은 이미지를 처리하는 VAE-based Deep SVDD와 DASVDD(Deep Autoencoding SVDD)를 벤치마킹하여 영상 처리에 적합하도록 재설계한 VAE-SVDD 모델의 입력 값으로 넣으면 outlier 지점을 파악하여 사고의 시점을 찾아낼 수 있다. 

모델의 학습 과정은 사전 학습한 3D CNN모델에 CCD(Car Crash Dataset)의 사고 영상 데이터를 추가 학습시켜 가중치를 미세조정한다. CCD는 환경 속성(낮/밤, 눈/비/좋은 날씨 조건), 자가 차량 관련 여부, 사고 참가자 및 사고 이유 설명을 포함한 다양한 사고 주석을 가지고 있어 기존 데이터 세트와 구별된다. 



3D CNN 모델을 선정하는 과정에서 실험을 통해 모델을 결정하였다. 결과는 다음과 같다.


Top-1에서 최고 성능 모델(X3D)과 최저 성능 모델(S3D)의 정확도 차이가 15%로 가장 많이 벌어졌고,
두 번째로 좋은 성능을 보이는 모델(R(2+1)D)과 최고 성능 모델(X3D)간의 차이는 0.4%로 가장 좁았다.
또한, 정확도와 정밀도, 재현율, F1 Score 모든 면에서 X3D, R(2+1)D, I3D, S3D 순으로 높은 결과를 보였다.


그림 1, 2, 3, 4는 각 모델에서의 예측 결과에 대한 ROC 그래프를 나타낸다. 400개 클래스에 대한 ROC
곡선(회색)과 이들의 평균(적색)을 포함하며, 각 클래스 별 ROC 곡선을 모두 구한 뒤 산출한 평균 ROC
곡선에서 산정되는 AUROC를 사용하여 모델 간 성능을 정량적으로 평가하였다. 모델의 성능을 측정하기 위한 기준으로 AUROC를 적용하였고 표4를 바탕으로 각 모델을 비교하였을 때 R(2+1)D의 성능이 가장 좋은 것으로 확인되었다.




3.2. 설계의 타당성
// Conceptual Design을 검증하기 위한 시뮬레이션이나 prototype을 만들어 실행시킨 결과를 자세히 기술 (각종 정량적 데이터 및 실행 결과를 사진 및 도표와 함께 제시할 것, 인공지능관련 과제의 경우는, 계산 시간과 인식 정확도 실험결과를 표로 정리해서 제시해야 함. (김민채, 1~2페이지 분량)
데이터 : 1개의 비디오당 총 50 frame으로, 1초당 10frame씩 재생하는 CCD (Car Crash Dataset)을 이용하였다.

Feature 추출 기법 : CCD pre-trained weight가 존재하는 3D CNN인 C3D, I3D, R(2+1)D를 이용하였다.

Feature 추출 과정
C3D
모델 구조는 5개의 convolution layer와 5개의 fully conneted layer로, Input clip size는 3 * 16 * 112 * 112이고 Output clip size는 1 * 4096이다.
I3D
모델 구조는 Two-Stream Inflated 3D Convolution netwrks으로, C3D에 RGB + optical flow를 적용했다.  필터의 dimension 늘리고, weight를 1/N으로 만들었으며, 여러 크기의 convolution을 병렬로 연결한 Inceptrion Block을 사용했다. feature 추출로 Normal(3000개), Car Crash(1500개)영상을 사용했으며, 50 frame에서 가운데 fram을 겹쳐서 앞뒤로 32개의 frame씩 추출했다. 버려지는 frame 없이 총 2개의 segment가 추출되었다.

R(2+1)D
모델 구조는 residual learning(잔류 학습)을 Convolution filter을 공간과 시간으로 분류한 3D CNN에 적용시켰다. 새로운 spatiotemporal block을 사용하였다. I3D와 마찬가지로 50 frame에서 가운데 fram을 겹쳐서 앞뒤로 32개의 frame씩 추출했다. 버려지는 frame 없이 총 2개의 segment가 추출되었다. ccd_crash.txt파일에 적혀있는 파일 영상들에 대해 feature extraction을 수행했다.


Outlier detection model 세팅
LOF
LOF(Local Outlier Factor)는 근처 데이터의 상대적인 밀도를 고려하여 밀집 지역에서 밀도 관점에서 봤을 때 급격한 감소가 이루어지면 이상점이라 판단하는 모델이다. 
IForest
IForest(Isolation forest)는 데이터를 Decision tree(의사결정나무)형태로 표현하여 정상값을 분리하기 위해 Decision Tree를 깊게 타고 내려가야 하고, 반대로 이상값은 Decision Tree 상단부에서 분리할 수 있다는 것을 이용하여 데이터가 Decision Tree의 얼마정도 깊이에 있는지에 따라서 이상점을 분리하는 모델이다.
세팅
실험에서는 normal 2970개, crash 30개로 train을 하였고, normal 95개, crash 5개로 test하였다. N_neighbors/N_estimators 범위를 지정하여 연속으로 실험을 진행하였다. 측정 값으로는 Test에서만 AUROC, Precision, Recall, f1 score을 사용하였다.


Outlier detection 결과
LOF
C3D


I3D

IForest
C3D

I3D

Umap을 이용한 Visualization





4.  프로젝트 추진
4.1. 수행한 작업

Kinetics400 데이터를 이용하여 3D CNN 모델 적용 시켜보기
사용 알고리즘
I3D
S3D
R(2+1)D
X3D



4.2 팀 협력 방안과 팀원별 역할

[팀 협력 방안]
data labeling : 6명 모두 
feature extraction : 3명
출력한 feature extraction excel 파일을 outlier 팀에게 전달
outlier  : 3명 


[팀원간의 역할 분배]
Feature Extraction 및 3D CNN (3명)





→ Top-1/5/10 class 추출


→ 정답인 영상은 1, 틀린 영상은 0


→ feature extraction 결과값 csv 파일에 저장



Outlier (3명)






→ auroc, accuracy, precision, recall 값 추출


→ Iforest를 이용하여 precision, recall, f1값 출력

→ visualization : umap 사용


5. 필요한 자원
5.1.	구현 및 실험에 예상되는 소요 부품 리스트	
서버


5.2.	프로젝트 수행에 필요한 예산 내역
서버 유지 관리 및 보수 비용

6. 첨부
6.1.	인용 자료 및 참고 문헌
3D CNN
[C3D] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, & Manohar Paluri. (2014). Learning Spatiotemporal Features with 3D Convolutional Networks. Cornell University - ArXiv. https://doi.org/10.48550/arxiv.1412.0767
[I3D] Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2017.502
[P3D] Qiu, Z., Yao, T., & Mei, T. (2017). Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks. 2017 IEEE International Conference on Computer Vision (ICCV). https://doi.org/10.1109/iccv.2017.590
[R2+1D] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, & Manohar Paluri. (2017). A Closer Look at Spatiotemporal Convolutions for Action Recognition. Cornell University - ArXiv. https://doi.org/10.48550/arxiv.1711.11248
[S3D] Xie, S., Sun, C., Huang, J., Tu, Z., & Murphy, K. (2018). Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification. Computer Vision – ECCV 2018, 318–335. https://doi.org/10.1007/978-3-030-01267-0_19
[SlowFast] Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). SlowFast Networks for Video Recognition. 2019 IEEE/CVF International Conference on Computer Vision (ICCV). https://doi.org/10.1109/iccv.2019.00630
[X3D] Feichtenhofer, C. (2020). X3D: Expanding Architectures for Efficient Video Recognition. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr42600.2020.00028

Outlier detection
[iForest] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. 2008 Eighth IEEE International Conference on Data Mining. https://doi.org/10.1109/icdm.2008.17
[LOF] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF. ACM SIGMOD Record, 29(2), 93–104. https://doi.org/10.1145/335191.335388

Dataset
[A3D] Yao, Y., Xu, M., Wang, Y., Crandall, D. J., & Atkins, E. M. (2019). Unsupervised Traffic Accident Detection in First-Person Videos. 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). https://doi.org/10.1109/iros40897.2019.8967556
[CCD] Bao, W., Yu, Q., & Kong, Y. (2020). Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning. Proceedings of the 28th ACM International Conference on Multimedia. https://doi.org/10.1145/3394171.3413827
[DOTA] Xia, G. S., Bai, X., Ding, J., Zhu, Z., Belongie, S., Luo, J., Datcu, M., Pelillo, M., & Zhang, L. (2018). DOTA: A Large-Scale Dataset for Object Detection in Aerial Images. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. https://doi.org/10.1109/cvpr.2018.00418
[Kinetics] Andrew Zisserman, Joao Carreira, Karen Simonyan, Will Kay, Brian Hu Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, & Mustafa Suleyman. (2017a). The Kinetics Human Action Video Dataset. ArXiv: Computer Vision and Pattern Recognition.

6.2. 실험 데이터, 수식 전개, 증명 등 세부 기술적인 사항들
실험은 Kinetics-400 데이터셋으로 진행
실험에 사용된 4가지 3D CNN 모델들은 모두 Kinetics-400 dataset으로 pretrained 됨
pretrained model은 34342개의 동영상을 사용함
실험에 사용된 장비
AMD Ryzen 9 3900XT 12-Core Processor
128GB RAM DDR4
GeForce RTX 3090 GPU
실험 방법
I3D와 S3D는 64개의 프레임을 한 세그먼트로 하여 256x256 픽셀 단위로 가져와 224x224 크기로 잘라서 사용
R(2+1)D는 32개의 프레임이 한 세그먼트로 구성되어 128x171 픽셀 크기로 가져와 가운데를 112x112 픽셀로 잘라서 사용
X3D는 총 프레임 수를 1 / 시간축 활성화 크기 비율로 자른 것을 한 세그먼트로 하여 112x112 픽셀로 조정하여 사용
수식 전개
Accuracy: 
Recall

Precision

F1Score

6.3. 기타 첨부자료 
// 이 프로젝트결과를 토대로 <캡스톤디자인>으로 이어서 프로젝트를 수행하기 위해서 알아야 하는 자세한 기술적 도움 정보들

