import numpy as np
import os
import shutil
fpath = '/data/home/cunyuangao/AIC20_ReID/image_query'
images_dir = '/data/home/cunyuangao/Project/kmeans/test2train_image'
with open('labels.txt','r') as f:
    count = 1
    car_id = 18291
    for f in f.readlines()[18290:]:
        pid = int(f.strip())
        pid = pid + 1763
        cam = np.random.randint(1,40)
        fname = ('{:08d}_{:03d}_{:06d}.jpg'
                 .format(pid, cam, car_id))
        ori_image = ('{:06d}.jpg'.format(count))
        count += 1
        car_id += 1
        shutil.copy(os.path.join(fpath,ori_image), os.path.join(images_dir, fname))