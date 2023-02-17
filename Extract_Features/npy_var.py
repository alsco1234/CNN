# Show feature shape

import csv
import numpy as np
from contextlib import nullcontext
import os
import shutil
import glob
from PIL import Image

f = '../s3d_0217ver_test.npy'
tmp = np.load(f, allow_pickle=True)

#tmp = tmp.tolist()
# for j in tmp:
#     print(i,': ',j)
#tmp['s3d'][0]로 뽑아

# if npz, instead of npy : using tmp['arr_1']

print(tmp.shape)
print(tmp)