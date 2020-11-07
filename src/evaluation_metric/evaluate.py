import numpy as np
import cv2
from os.path import join
import random
import math 
import re 
from datetime import datetime

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input

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
        if g in mask:
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


def search(files, db_path, mask_path, video_path, params):
    memory_graph = MemoryGraph(db_path, params)
    cnn = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    orb = cv2.ORB_create(nfeatures=100000, fastThreshold=7)

    for file in files:
        search_file(file, memory_graph, cnn, orb, mask_path, video_path, params)


## get the k nearest neighbors for the given input image
def search_file(image_file, mask, memory_graph, cnn, orb, params):
    print("search", file)

    pil_image = Image.open(image_file).convert('RGB')
    image = np.array(pil_image)

    g_pos = [None for _ in range(params["search_walker_count"])]
    pos = [None for _ in range(params["search_walker_count"])]
    adj = [False for _ in range(params["search_walker_count"])]
    walk_t = [0 for _ in range(params["search_walker_count"])]
    cluster_feats = [[] for _ in range(params["search_walker_count"])]
    cluster_positions = [[] for _ in range(params["search_walker_count"])]
    cluster_patches = [[] for _ in range(params["search_walker_count"])]

    observation_ids = set()

    done = False

    for t in range(params["search_max_frames"]):
        if t % params["search_walk_length"] == 0:
            print("frame", t)

        video_shape = image.shape

        kp_grid = key_point_grid(orb, image, mask, params["grid_margin"], params["stride"])

        for i in range(params["search_walker_count"]):
            g_pos[i], pos[i], adj[i] = next_pos(kp_grid, video_shape, g_pos[i], walk_t[i], params["search_walk_length"], params["stride"])
            if t == 0:
                adj[i] = True
            if adj[i]:
                walk_t[i] += 1
            else:
                walk_t[i] = 0

        patches = extract_windows(image, pos, params["window_size"])
        windows = patches.astype(np.float64)

        preprocess_input(windows)
        feats = cnn.predict(windows)
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



    
    print("")
    print(datetime.now())
    print(params)
    print(object_name)
    print("len(observations)", len(observations))
    print("")



files = [
    '001_apple.mp4',
    '002_apple.mp4',
    '003_apple.mp4',
    '004_bear.mp4',
    '005_bear.mp4',
    '006_bear.mp4',
    '007_brush.mp4',
    '008_brush.mp4',
    '009_brush.mp4',
    '010_carrot.mp4',
    '011_carrot.mp4',
    '012_carrot.mp4',
    '013_chain.mp4',
    '014_chain.mp4',
    '015_chain.mp4',
    '016_clippers.mp4',
    '017_clippers.mp4',
    '018_clippers.mp4',
    '019_cologne.mp4',
    '020_cologne.mp4',
    '021_cologne.mp4',
    '022_cup.mp4',
    '023_cup.mp4',
    '024_cup.mp4',
    '025_flowers.mp4',
    '026_flowers.mp4',
    '027_flowers.mp4',
    '028_hanger.mp4',
    '029_hanger.mp4',
    '030_hanger.mp4',
    '031_ketchup.mp4',
    '032_ketchup.mp4',
    '033_ketchup.mp4',
    '034_notebook.mp4',
    '035_notebook.mp4',
    '036_notebook.mp4',
    '037_opener.mp4',
    '038_opener.mp4',
    '039_opener.mp4',
    '040_pepper.mp4',
    '041_pepper.mp4',
    '042_pepper.mp4',
    '043_rock.mp4',
    '044_rock.mp4',
    '045_rock.mp4',
    '046_shorts.mp4',
    '047_shorts.mp4',
    '048_shorts.mp4',
]

mask_path = "../../media/tabletop_objects/masks/"
video_path = "../../media/tabletop_objects/videos/"
db_path = "../../data/table_objects_j.db"

search(files, db_path, mask_path, video_path, PARAMETERS)