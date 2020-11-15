from glob import glob
from vgg16_window_walker_lib_images import build_graph, PARAMETERS
import random

db_path = "../../data/variations_test.db"
image_files = glob("dataset_100/train/*/*.jpg")
random.shuffle(image_files)

print('len(image_files)', len(image_files))

build_graph(db_path, image_files, PARAMETERS)
