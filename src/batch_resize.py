from glob import glob
import cv2
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool
import os 


root_dir = 'open-images-dataset''
image_paths = glob('/data/open-images-dataset/*/*.jpg')
shortest_dim_size = 348 # 2*(128 + 2*22 + 2*1)


def resize_image(tuple_value):
    index, image_path = tuple_value

    if index % 1000 == 0:
        print(index, image_path)

    image = cv2.imread(image_path)

    if image.shape[1] > image.shape[0]:
        image = cv2.resize(image, (int(round(image.shape[1]/image.shape[0] * shortest_dim_size)), shortest_dim_size))
    else:
        image = cv2.resize(image, (shortest_dim_size, int(round(image.shape[0]/image.shape[1] * shortest_dim_size))))

    sub_path = image_path[image_path.index(root_dir)+len(root_dir):]
    new_image_path = f'mini_{root_dir}{sub_path}'
    os.makedirs(os.path.dirname(new_image_path), exist_ok = True) 

    cv2.imwrite(new_image_path, image)
 

print("Starting", len(image_paths))
pool = ThreadPool(4)
pool.imap(resize_image, enumerate(image_paths))

