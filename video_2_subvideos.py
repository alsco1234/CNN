# for x3d, where is time axis?

import cv2
vidcap = cv2.VideoCapture('../../extra/sun1.mp4')
success,image = vidcap.read()

count = 1
success = True
frame_array = []

pathOut = '../../extraresults/'
fps = 10
size = (1920, 1080) #width, height

while success:
  success,image = vidcap.read()
  cv2.imwrite("../../extraresults/sun1/" + str(count) + ".jpg", image)
  print("saved image " + str(count) + ".jpg")
  
  if cv2.waitKey(10) == 27:                    
      break
  count += 1

  frame_array.append(image)
  # 16개씩 저장
  if count%16 == 0:
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    
    out.release()
    frame_array = []

# # https://thinking-developer.tistory.com/61
# #라이브러리 호출
# import cv2
# import os
     
# filepath = '../../extra/sun1.mp4'
# video = cv2.VideoCapture(filepath) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

# if not video.isOpened():
#     print("Could not Open :", filepath)
#     exit(0)

# fps = video.get(cv2.CAP_PROP_FPS)

# #프레임을 저장할 디렉토리를 생성
# try:
#     if not os.path.exists(filepath[:-4]):
#         os.makedirs(filepath[:-4])
# except OSError:
#     print ('Error: Creating directory. ' +  filepath[:-4])

# count = 0

# while(video.isOpened()):
#     ret, image = video.read()
#     if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
#         cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % count, image)
#         print('Saved frame number :', str(int(video.get(1))))
#         count += 1
        
# video.release()
     