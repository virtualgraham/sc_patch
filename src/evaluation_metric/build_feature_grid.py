from glob import glob
import math
from tqdm import tqdm
import numpy as np
from PIL import Image
import pathlib

from MoCoFeatureExtractor import MoCoFeatureExtractor

version = 'c'

window_size_outer = 288
window_size_inner = 144

stride = 72

image_scale = 1

feature_dim = 2048

cnn = MoCoFeatureExtractor(params_path='/home/ubuntu/moco_v2_800ep_pretrain.pth.tar')

image_files = glob("/home/ubuntu/dataset_1000/train/*/*.jpg")



def extract_windows(frame, pos, window_size):
    windows = np.empty((len(pos), window_size, window_size, 3), dtype=np.uint8)

    for i in range(len(pos)):
        windows[i] = extract_window(frame, pos[i], window_size)

    return windows


def extract_window(frame, pos, window_size):
    half_w = window_size/2.0

    top_left = [int(round(pos[0]-half_w)), int(round(pos[1]-half_w))]
    bottom_right = [top_left[0]+window_size, top_left[1]+window_size]

    return frame[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]


for idx, image_file in tqdm(enumerate(image_files), total=len(image_files)):
    
    pil_image = Image.open(image_file).convert('RGB')
    pil_image = pil_image.resize((int(round(pil_image.size[0] * image_scale)), int(round(pil_image.size[1] * image_scale))))
    image = np.array(pil_image)
    
    if image.shape[0] < window_size_outer * 2 or image.shape[1] < window_size_outer * 2 or image.shape[0] > 1024 or image.shape[1] > 1024:
        continue

    margin = window_size_outer-stride
    grid_shape = (math.floor((image.shape[0] - margin) / stride), math.floor((image.shape[1] - margin) / stride))
    offsets = (round((image.shape[0] - grid_shape[0] * stride)/2), round((image.shape[1] - grid_shape[1] * stride)/2))

    points = [(offsets[0]+y*stride+stride/2,offsets[1]+x*stride+stride/2) for y in range(grid_shape[0]) for x in range(grid_shape[1])]

    patches_outer = extract_windows(image, points, window_size_outer)
    windows_outer = patches_outer.astype(np.float64)
    
    patches_inner = extract_windows(image, points, window_size_inner)
    windows_inner = patches_inner.astype(np.float64)
    
    try:
        feats_outer = cnn.evalRGB(windows_outer)
        feats_inner = cnn.evalRGB(windows_inner)
    except:
        print("ERROR cnn.evalRGB", image, image.shape, windows_outer.shape, windows_inner.shape)
        raise

    feat_grid_outer = feats_outer.reshape((grid_shape[0], grid_shape[1], feature_dim))
    feat_grid_inner = feats_inner.reshape((grid_shape[0], grid_shape[1], feature_dim))
    
    path_parts = image_file.split('/')
    image_id = path_parts[-1].split('.')[0]
    image_class = path_parts[-2]
    
    pathlib.Path(f'feat_grids_{window_size_outer}_{window_size_inner}_{stride}_{version}/{image_class}').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f'feat_grids_{window_size_outer}_{window_size_inner}_{stride}_{version}/{image_class}/{image_id}.npz', outer=feat_grid_outer, inner=feat_grid_inner)