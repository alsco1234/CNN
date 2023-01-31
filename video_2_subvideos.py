# import cv2
# vidcap = cv2.VideoCapture('../../extra/sun1.mp4')
# success,image = vidcap.read()

# count = 1
# success = True

# subnum = 0

# while success:
#   success,image = vidcap.read()
#   cv2.imwrite("../../extraresults/%d/sun1_%d.jpg" % subnum % count, image)
#   print("saved image %d.jpg in %d folder." % count % subnum)
  
#   if cv2.waitKey(10) == 27:                    
#       break
#   count += 1

#   if count%16 == 0:
#      subnum +=1

# https://thinking-developer.tistory.com/61
#라이브러리 호출
import cv2
import os
     
filepath = '../../extra/sun1.mp4'
video = cv2.VideoCapture(filepath) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

if not video.isOpened():
    print("Could not Open :", filepath)
    exit(0)

fps = video.get(cv2.CAP_PROP_FPS)

#프레임을 저장할 디렉토리를 생성
try:
    if not os.path.exists(filepath[:-4]):
        os.makedirs(filepath[:-4])
except OSError:
    print ('Error: Creating directory. ' +  filepath[:-4])

count = 0

while(video.isOpened()):
    ret, image = video.read()
    if(int(video.get(0.1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 0.1초마다 추출
        cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % count, image)
        print('Saved frame number :', str(int(video.get(1))))
        count += 1
        
video.release()
     