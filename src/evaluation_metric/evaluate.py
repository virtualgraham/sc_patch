import numpy as np
import cv2
from os.path import join
import random
import math 
import re 
from datetime import datetime


from ShufflePatchModel import ShufflePatchFeatureExtractor

# from tensorflow.keras.applications import vgg16
# from tensorflow.keras.applications.vgg16 import preprocess_input

from vgg16_window_walker_lib_images import color_fun, extract_windows, extract_window, extract_object, get_rad_grid, MemoryGraph, extract_window_pixels, PARAMETERS

from itertools import chain



def key_point_grid(orb, frame, mask, grid_margin, stride):

    grid_width = math.floor((frame.shape[0] - grid_margin*2) / stride)
    grid_height = math.floor((frame.shape[1] - grid_margin*2) / stride)

    grid_offset_x = ((frame.shape[0] - grid_margin*2) % stride)/2.0 + grid_margin
    grid_offset_y = ((frame.shape[1] - grid_margin*2) % stride)/2.0 + grid_margin

    # object_grid_locations = set()

    #print("grid_width", grid_width, "grid_height", grid_height)
    # for x in range(grid_width):
    #     for y in range(grid_height):
    #         p = (grid_offset_x + x * stride + 0.5 * stride, grid_offset_y + y * stride + 0.5 * stride)
    #         w = extract_window(frame, p, stride)
    #         if extract_object(w, stride) is not None:
    #             object_grid_locations.add((x, y))
    
    #print("len(object_grid_locations)", len(object_grid_locations))
    kp = orb.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    #print("len(kp)", len(kp))

    grid = dict()

    for k in kp:
        p = (k.pt[1],k.pt[0])
        g = (int(math.floor((p[0]-grid_offset_x)/stride)), int(math.floor((p[1]-grid_offset_y)/stride)))

        ### TODO: implement if g in mask
        if True # g in mask:
            if g in grid:
                grid[g].append(p)
            else:
                grid[g] = [p]

    return grid


def first_pos(kp_grid):
    loc = random.choice(list(kp_grid.keys()))
    return loc, random.choice(kp_grid[loc])


def next_pos(kp_grid, shape, g_pos, walk_t, walk_length, stride):
    
    if (g_pos is not None) and walk_t < (walk_length-1):

        for rad in range(1, 3):
            rad_grid = get_rad_grid(g_pos, rad, shape, stride)

            if len(rad_grid) == 0:
                print("frame empty?")
                break

            random.shuffle(rad_grid)

            for loc in rad_grid:
                if loc in kp_grid:
                    return loc, random.choice(kp_grid[loc]), True
    
    loc, pos = first_pos(kp_grid)
    return loc, pos, False


def search(image_files, mask_files, db_path, params):
    memory_graph = MemoryGraph(db_path, params)

    cnn = ShufflePatchFeatureExtractor("/Users/racoon/Desktop/variation_2b_migrated_0135_0.001_1.4328_63.80.pt")
    # cnn = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    orb = cv2.ORB_create(nfeatures=100000, fastThreshold=7)

    return [search_file(image_file[i], mask_file[i], memory_graph, cnn, orb, params) for i in range(len(image_files))]


## get the k nearest neighbors for the given input image
def search_file(image_file, mask_file, memory_graph, cnn, orb, params):
    print("search", file)

    pil_image = Image.open(image_file).convert('RGB')
    image = np.array(pil_image)

    pil_mask = Image.open(mask_file).convert('1')
    mask = np.array(pil_mask)

    g_pos = [None for _ in range(params["search_walker_count"])]
    pos = [None for _ in range(params["search_walker_count"])]
    adj = [False for _ in range(params["search_walker_count"])]
    walk_t = [0 for _ in range(params["search_walker_count"])]
    cluster_feats = [[] for _ in range(params["search_walker_count"])]
    cluster_positions = [[] for _ in range(params["search_walker_count"])]
    cluster_patches = [[] for _ in range(params["search_walker_count"])]

    image_shape = image.shape
    kp_grid = key_point_grid(orb, image, mask, params["grid_margin"], params["stride"])


    observation_ids = set()

    done = False

    for t in range(params["search_max_frames"]):
        if t % params["search_walk_length"] == 0:
            print("frame", t)

        for i in range(params["search_walker_count"]):
            g_pos[i], pos[i], adj[i] = next_pos(kp_grid, image_shape, g_pos[i], walk_t[i], params["search_walk_length"], params["stride"])
            if t == 0:
                adj[i] = True
            if adj[i]:
                walk_t[i] += 1
            else:
                walk_t[i] = 0

        patches = extract_windows(image, pos, params["window_size"])
        windows = patches.astype(np.float64)

        feats = cnn.evalRGB(windows)

        # preprocess_input(windows)
        # feats = cnn.predict(windows)

        feats = feats.reshape((windows.shape[0], 512))
        
        for i in range(params["search_walker_count"]):
            cluster_feats[i].append(feats[i])
            cluster_positions[i].append(pos[i])
            cluster_patches[i].append(patches[i])


        for i in range(params["search_walker_count"]):
            if (not adj[i] or done) and len(cluster_feats[i]) > 0:

                ########
                
                similar_clusters = memory_graph.search_group_foo(cluster_feats[i], params)
                node_ids = set(chain.from_iterable(similar_clusters))
                observation_ids.update(memory_graph.observations_for_nodes(node_ids))
                
                ########

                cluster_feats[i] = []
                cluster_positions[i] = []
                cluster_patches[i] = []
        
        if done:
            break

    print(image_file)

image_files = glob()
mask_files = glob()
db_path = "../../data/table_objects_j.db"

search(image_files, mask_files, db_path, PARAMETERS)