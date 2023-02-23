""" n번째 프레임만 추출"""

import os
 
folder_dir = 'projects/vode/DEC_origins'
folder_list = os.listdir(folder_dir)

for foldername in folder_list:
    path_dir = folder_dir + "/" + foldername
    file_list = os.listdir(path_dir)

    for filename in file_list:
        if (int(filename[0:4]) + 3) % 3 !=0:
            os.remove(path_dir + "/" + filename)