# Show feature shape

import numpy as np
from PIL import Image

f = '../../27.npy'
tmp = np.load(f, allow_pickle=True)

tmp = tmp.tolist()
#for j in tmp:
#    print(i,': ',j)
#tmp['s3d'][0]로 뽑아

# if npz, instead of npy : using tmp['arr_1']

#print(tmp.shape)
print(tmp)