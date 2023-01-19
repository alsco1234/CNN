import csv
import numpy as np
from contextlib import nullcontext
import os
import shutil
import glob
from PIL import Image

f = '/projects/vode/ccd_x3d_npz/normal/00001_x3d.npz'
tmp = np.load(f, allow_pickle=True)

tmp = tmp['arr_0'].tolist()
# for j in tmp:
#     print(i,': ',j)
#tmp['s3d'][0]로 뽑아

print(tmp.shape())