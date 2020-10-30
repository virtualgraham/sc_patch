from glob import glob
import os
from random import sample 
from shutil import copy

sample_image_paths = sample(glob('/data/open-images-dataset/train.*.jpg'), 1000)

dst = '/data/open-images-sample'
os.mkdir(dst)
for path in sample_image_paths:
    copy(path, dst)