import cv2
import os
from os import path
from pathlib import Path

filepath = '../../../projects/vode/CCD/Normal/'

file_names = os.listdir('../../../projects/vode/CCD/Normal/')
file_list = list(file_names)

for file in file_list:
    count = 0
    file_num = 0
    str = file.split('.')
    vidcap =  cv2.VideoCapture(filepath + file)
    print(filepath + file)

    while(vidcap.isOpened()):

        ret, image = vidcap.read()
        '''
        if count % 2 == 0:
            dir_path = Path("../../../projects/vode/data/zeromin/ccd_c3d/test/" + str[0])
            if dir_path.is_dir() == True:
                cv2.imwrite("../../../projects/vode/data/zeromin/ccd_c3d/test/" + str[0] + "/" + "%d.jpg" %file_num, image)
                file_num += 1
                #print('Saved framed%d.jpg'%count)
            else:
                os.makedirs("../../../projects/vode/data/zeromin/ccd_c3d/test/" + str[0])
                cv2.imwrite("../../../projects/vode/data/zeromin/ccd_c3d/test/" + str[0] + "/" + "%d.jpg" %file_num, image)
                file_num += 1
                #print('Saved framed%d.jpg'%count)
        '''
        if count >= 34:
            dir_path = Path("../../../projects/vode/data/zeromin/ccd_c3d/normal/YoYo/" + str[0])
            if dir_path.is_dir() == True:
                cv2.imwrite("../../../projects/vode/data/zeromin/ccd_c3d/normal/YoYo/" + str[0] + "/" + "%d.jpg" %file_num, image)
                file_num += 1
                #print('Saved framed%d.jpg'%count)
            else:
                os.makedirs("../../../projects/vode/data/zeromin/ccd_c3d/normal/YoYo/" + str[0])
                cv2.imwrite("../../../projects/vode/data/zeromin/ccd_c3d/normal/YoYo/" + str[0] + "/" + "%d.jpg" %file_num, image)
                file_num += 1
                #print('Saved framed%d.jpg'%count)


        count += 50
        # vidcap.releas()
        if file_num == 16:
            print("Saved " + file + " Done")
            break

