# feature reshape (1, -1)

import os
import numpy as np

path = '/home/s21900395/video_features/crash_32_18_r21d/r21d/'

for i in os.listdir(path):
    data = np.load(path+i)
    reshape_data = data.reshape(1,-1)
    np.save('/home/s21900395/video_features/r21d_crash_reshape/'+i, reshape_data)
    print('file ', i, ' saved')

