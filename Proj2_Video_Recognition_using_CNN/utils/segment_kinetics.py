import cv2
import os
from os import path
from pathlib import Path

# filepath = '../../../projects/vode/CCD/Normal/000004.mp4'
filepath = '/Users/kimminchae/Desktop/CNN/k400/'
# vidcap = cv2.VideoCapture(filepath)

file_names = os.listdir('../../../projects/vode/data/kinetics400/test/')
file_list = list(file_names)

for file in file_list:
    video_names = os.listdir('/Users/kimminchae/Desktop/CNN/k400/' + file + '/')
    video_list = list(video_names)
    for video in video_list:
        count = 0
        file_num = 0
        str = video.split('.')
        vidcap = cv2.VideoCapture(filepath + file + '/' + video)
        print(filepath + file + '/' + video)
        while(vidcap.isOpend()):

            ret, image = vidcap.read()

            if count % 2 == 0:
                dir_path = Path("../../../projects/vode/data/video_segment/kinetics400_100/test/" + file)
                if dir_path.is_dir() == True:
                    video_path = Path("../../../projects/vode/data/video_segment/kinetics400_100/test/" + file + '/' + str[0])
                    if video_path.is_dir() == True:
                        cv2.imwrite("../../../projects/vode/data/video_segment/kinetics400_100/test/" + file + '/' + str[0] + "/" + "%d.jpg" %file_num, image)
                        file_num += 1
                    else:
                        os.makedirs("../../../projects/vode/data/video_segment/kinetics400_100/test/" + file + '/' + str[0])
                        cv2.imwrite("../../../projects/vode/data/video_segment/kinetics400_100/test/" + file + '/' + str[0] + "/" + "%d.jpg" %file_num, image)
                        file_num += 1
                else:
                    os.makedirs("../../../projects/vode/data/video_segment/kinetics400_100/test/" + file)
                    os.makedirs("../../../projects/vode/data/video_segment/kinetics400_100/test/" + file + '/' + str[0])
                    cv2.imwrite("../../../projects/vode/data/video_segment/kinetics400_100/test/" + file + '/' + str[0] + "/" + "%d.jpg" %file_num, image)
                    file_num += 1
            count += 1

            if file_num == 100:
                print("Saved " + file + "Done")
                break
