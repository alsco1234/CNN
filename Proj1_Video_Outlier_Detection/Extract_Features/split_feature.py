"""
While Using CCD(Car Crash Dataset), Crash will accure about 1/3 frame
So, this code will split crash 50 frame into 2 + 16 + 16 + 16, get real crash segment
"""

import os
import numpy as np
import csv

dir_path = "/home/s21900395/new_video_features/video_features/16_crash_r21d/r2plus1d_18_16_kinetics" # feature 경로 
#f = open("/home/s21900395/new_video_features/video_features/Crash_Table.csv", "r")
#reader = csv.reader(f)

def createFolder(dir): # 폴더 존재하지 않으면 생성 
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: cannot make a directory. ' + dir)

for (root, dictionalries, files) in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(root, file) # 파일 경로
        video_num = file[:6] # 비디오 번호 

        data = np.load(file_path) # numpy 하나씩 불러오기

        #print('video_num:', int(video_num.lstrip("0")))

        f = open("/home/s21900395/new_video_features/video_features/Crash_Table.csv", "r") # csv 파일 경로 
        reader = csv.reader(f)
        for line in reader: # csv 파일 한 줄 씩 읽어오기
            #print('line[0]: ', line[0])
            #print('video_num: ', int(video_num.lstrip("0")))
            if line[0] == video_num.lstrip("0"): # csv에서 video 번호 찾기
                """
                print('line[0]:', line[0])
                print('video_num:', video_num.lstrip("0"))
                """
                for i in range(3):
                    label = 'normal'
                    arr = data[i, :] # numpy arr 하나씩 불러오기
                    if i == 0: # frame (2, 18) --> csv에서는 3부터 18까지 
                        if "1" in line[3:19]:
                            label = 'anomaly'
                    elif i == 1: # frame (18, 34) --> csv에서는 19부터 34까지
                        if "1" in line[19:35]:
                            label = 'anomaly'
                    else: # frame (34, 50) --> csv에서는 35부터 끝까지 
                        if "1" in line[35:]:
                            label = 'anomaly'

                    # 001 video의 1번째 row이면, /001/1.npy 로 저장됨
                    if label == 'normal': # 정상 segment이면 
                        print('normal')
                        createFolder("./crash/normal/" + video_num)
                        np.save("./crash/normal/" + video_num + "/" + str(i) + ".npy", arr) # normal
                    else:
                        print('anomaly')
                        createFolder("./crash/anomaly/" + video_num)
                        np.save("./crash/anomaly/" + video_num + "/" + str(i) + ".npy", arr) # anomaly
        f.close()