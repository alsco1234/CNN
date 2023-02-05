문제정의 
(Problem definition)
블랙박스 영상 분석 비용 축소를 목적으로 블랙박스 영상에 video outlier detection모델을 사용하는 자동화 기술을 개발을 목적으로 한다.

i) Input: CCD dataset의 영상 데이터와 교통상황 감시카메라 혹은 직접 수집한 블랙박스 데이터를 사용할 모델의 학습에 알맞은 형태로 전처리 후, input데이터로 사용한다.

ii) function: 영상 분류 3D CNN모델 중 최고의 성능을 내는 모델을 선택하여 pre-trained모델을 찾고, 교통 법규를 위반한 차량사고 영상 데이터에 대해 fine-tuning을 진행한 뒤, feature extraction과정을 수행하여 input으로 들어온 사고 영상에 대한 특징을 추출한다. 추출된 영상의 특징은 영상당 하나의 1차원 array형태로 outlier detection 모델의 input으로 들어가게 되고, outlier detection모델에서는 이러한 특징들은 분석하여 실제로 사고가 발생하였거나, 교통 위반이 감지된 지점을 찾게 된다.

iii) output: 3D-CNN모델로 부터 영상 내 특징이 추출되고, 추출된 특징이 outlier detection 모델을 거치게 되면 최종 output이 나오게 된다. output은 각 영상의 특징을 분석하여 계산된 이상치를 기반으로 실제로 사고 장면이나, 교통 위반 장면이라고 의심되는 부분에 대한 영상내 지점을 반환하게 된다.



