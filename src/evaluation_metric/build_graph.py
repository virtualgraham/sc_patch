from glob import glob
from vgg16_window_walker_lib_images import build_graph, PARAMETERS

db_path = "../../data/table_objects_i.db"
image_files = glob("dataset_100/*/*.jpg")

print('len(image_files)', len(image_files))

build_graph(db_path, image_files, PARAMETERS)
