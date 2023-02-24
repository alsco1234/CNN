# feature reshape to (1, -1)

import os
import numpy as np

# forder path
path = './img/'

for i in os.listdir(path):
    data = np.load(path+i)
    reshape_data = data.reshape(1,-1)
    np.save('img_reshape'+i, reshape_data)
    print('file ', i, ' saved')

