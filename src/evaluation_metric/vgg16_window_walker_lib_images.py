import math
import random
import time
import struct
import os.path
from os import listdir
from os.path import isfile, isdir, join, split, splitext
import json
from pathlib import Path
import threading
import re 
from itertools import chain
from collections import Counter

import numpy as np
import cv2
import hnswlib
import plyvel
from PIL import Image

import community_walk_graph as cwg

from ShufflePatchModel import ShufflePatchFeatureExtractor
# from tensorflow.keras.applications import vgg16
# from tensorflow.keras.applications.vgg16 import preprocess_input

class MemoryGraphWalker:
    def __init__(self, memory_graph, params):

        self.knn = params["knn"]
        self.accurate_prediction_limit = params["accurate_prediction_limit"]
        self.identical_distance = params["identical_distance"]
        self.distance_threshold = params["distance_threshold"]
        self.prediction_history_length = params["prediction_history_length"]
        self.history_community_matches = params["history_community_matches"]
        self.keep_times = params["keep_times"]
        self.prevent_similar_adjacencies = params["prevent_similar_adjacencies"]

        self.walk_trials = params["walk_trials"]
        self.member_portion = params["member_portion"]

        self.memory_graph = memory_graph

        self.last_ids = dict()
        self.history_nn = dict()
        self.last_feats = dict()
        self.last_obs = dict()

    def add_parrelell_observations(self, file, t, pos, adj, feats, patches):
        if self.memory_graph.index_count() >= self.knn:
            labels, distances = self.memory_graph.knn_query(feats, k = self.knn)
        else:
            labels = [[] for i in range(len(feats))]
            distances = [[] for i in range(len(feats))]

        # get all the labels less than threshold distance together in one list
        labels_merged = list(chain.from_iterable(labels))
        distances_merged = list(chain.from_iterable(distances))

        neighbor_nodes_merged = list(set([l for l,d in zip(labels_merged, distances_merged) if d <= self.distance_threshold]))

        community_cache_list = self.memory_graph.get_communities(neighbor_nodes_merged, walk_trials=self.walk_trials, member_portion=self.member_portion)

        community_cache = dict([(neighbor_nodes_merged[i], frozenset(community_cache_list[i])) for i in range(len(neighbor_nodes_merged))])

        return [self._add_observation(file, t, pos[i], adj[i], feats[i], patches[i], labels[i], distances[i], i, community_cache) for i in range(len(feats))]



    def _add_observation(self, file, t, pos, adj, feats, patch, labels, distances, walker_id, community_cache):
 
        stats = {"adj":adj}

        tm = TimeMarker(enabled=self.keep_times)

        if self.prevent_similar_adjacencies:
            if adj and walker_id in self.last_feats:
                d = self.memory_graph.distance(feats, self.last_feats[walker_id])
                if d <= self.distance_threshold:
                    stats["skipped"] = True
                    return self.last_ids[walker_id], self.last_obs[walker_id], stats

        stats["skipped"] = False

        observation = {"file":file, "t":t, "y":pos[0], "x":pos[1], "patch":patch}
        # if obj is not None: observation["o"] = obj
        observation_id = self.memory_graph.insert_observation(observation)

        tm.mark(s="insert_observation")

        
        if len(distances) > 0 and distances[0] <= self.distance_threshold: 
            neighbor_nodes = set([l for l,d in zip(labels, distances) if d <= self.distance_threshold])
        else:
            neighbor_nodes = set()



        stats["near_neighbors_count"] = len(neighbor_nodes)

        if len(distances) > 0:
            stats["nearest_neighbor"] = distances[0]

        tm.mark(l="find_correct_predictions", s="knn_query")

        accurate_predictions = set()
        skipped_accurate_predictions = set()
        evaluated_ids = set()



        if adj and walker_id in self.history_nn and len(neighbor_nodes) > 0:

            tm.mark(l="find_correct_predictions_inside")

            add_predicted_observations = set()
   
            ###################
            ###################

            tm.mark(l="build_accurate_predictions_set")
            
            

            for nn in neighbor_nodes:
                nn_community = community_cache[nn]
                
                bar = set() # near neighbors to recent history that are in the community of this neighbor of the current observation
                baz = 0 # number of recent history that have near neighbors in the community of this neighbor of the current observation
                for h in self.history_nn[walker_id]:
                    foo = nn_community.intersection(h)
                    if len(foo) > 0:
                        bar.update(foo)
                        baz += 1
                if baz >= self.history_community_matches:
                    if self.prevent_similar_adjacencies:
                        for b in bar:
                            if b not in accurate_predictions and b not in skipped_accurate_predictions:
                                d = self.memory_graph.distance(self.memory_graph.get_node(b)["f"], feats)
                                if d > self.distance_threshold:
                                    accurate_predictions.add(b)
                                    if len(accurate_predictions) >= self.accurate_prediction_limit:
                                        break
                                else:
                                    skipped_accurate_predictions.add(b)
                    else:
                        accurate_predictions.update(bar)

                    add_predicted_observations.add(nn)

                if len(accurate_predictions) >= self.accurate_prediction_limit:
                    break
            
            stats["adjacencies_skipped"] = len(skipped_accurate_predictions)

            tm.mark(si="build_accurate_predictions_set")

            ###################
            ###################

            tm.mark(l="add_predicted_observations")

            if len(add_predicted_observations) > 0:
                self.memory_graph.add_predicted_observations(add_predicted_observations, [observation_id]*len(add_predicted_observations))

            tm.mark(si="add_predicted_observations")

            stats["accurate_predictions"] = len(accurate_predictions)
        
            tm.mark(si="find_correct_predictions_inside")

        tm.mark(si="find_correct_predictions")

        if len(accurate_predictions) < self.accurate_prediction_limit:

            if len(distances) > 0 and (distances[0] < self.identical_distance):
                node_id = labels[0]
                stats["identical"] = True

            else:
                node_id = self.memory_graph.insert_node({"f":feats})

            self.memory_graph.add_integrated_observations([node_id], [observation_id])
    
            stats["adjacencies_inserted"] = 0

            if adj:
                insert_adjacencies = []

                if walker_id in self.last_ids and self.last_ids[walker_id] is not None :
                    stats["adjacencies_inserted"] += 1
                    insert_adjacencies.append((self.last_ids[walker_id], node_id))

                for a in accurate_predictions:
                    stats["adjacencies_inserted"] += 1
                    insert_adjacencies.append((a, node_id))

                self.memory_graph.insert_adjacencies(insert_adjacencies)
        else:
            node_id = None

        tm.mark(s="insert_node_and_adjacencies")

        #################
        # updating history
        #################
        
        if walker_id not in self.history_nn:
            self.history_nn[walker_id] = []

        h = self.history_nn[walker_id]
        h.append(frozenset(neighbor_nodes))
        if len(h) > self.prediction_history_length:
            h.pop(0)


        tm.mark(s="make_predictions")

        if self.keep_times:
            stats["time"] = tm.saved

        self.last_ids[walker_id] = node_id
        self.last_feats[walker_id] = feats
        self.last_obs[walker_id] = observation_id

        return node_id, observation_id, stats
        


        
MAX_KEY_VALUE = 18446744073709551615


class MemoryGraph:
    #def __init__(self, path, space='cosine', dim=512, max_elements=1000000, ef=100, M=48, rebuild_index=False):
    def __init__(self, path, params):
        self.space = params["space"]
        self.dim = params["dim"]
        self.max_elements = params["max_elements"]
        self.ef = params["ef"]
        self.M = params["M"]
        self.path = path
        self.open(params["rebuild_index"])

    def save(self):
        print("Saving Index")
        index_path = os.path.splitext(self.path)[0] + ".index"
        print("index_path", index_path)
        self.index.save_index(index_path)
        print("Index Saved")

    def close(self):
        self.save()
        self.db.close()
        self.graph = None
        self.index = None
        self.db = None

    def open(self, rebuild_index):
        self.db = plyvel.DB(self.path, create_if_missing=True)

        self.graph = cwg.new_graph()

        index_path = os.path.splitext(self.path)[0] + ".index"
        print("index_path", index_path)
        self.index = hnswlib.Index(space=self.space, dim=self.dim) 

        if os.path.isfile(index_path) and not rebuild_index:
            print("MemoryGraph: loading index")
            self.index.load_index(index_path)
            self.index.set_ef(self.ef)
            self.load_all_node_ids()
        else:
            print("MemoryGraph: building index")
            self.index.init_index(max_elements=self.max_elements, ef_construction=self.ef * 2, M=self.M)
            self.index.set_ef(self.ef)
            self.load_all_nodes()
            if cwg.len(self.graph) > 0:
                self.save()
        
        self.load_all_edges()

        print("MemoryGraph:", self.index.get_current_count(), "nodes")


    def load_all_node_ids(self):
        start = MemoryGraph.node_key(0)
        stop = MemoryGraph.node_key(MAX_KEY_VALUE)
        for key in self.db.iterator(start=start, stop=stop, include_value=False):
            cwg.add_node(self.graph, MemoryGraph.decode_node_key(key))


    def load_all_nodes(self):
        start = MemoryGraph.node_key(0)
        stop = MemoryGraph.node_key(MAX_KEY_VALUE)

        n = 0
        feats = []
        ids = []

        for key, value in self.db.iterator(start=start, stop=stop):
            node = MemoryGraph.decode_node(value) 
            node["id"] = MemoryGraph.decode_node_key(key)

            cwg.add_node(self.graph, node["id"])
            
            feats.append(node["f"])
            ids.append(node["id"])

            n += 1
            if n % 1000 == 0:
                print(n)
                self.index.add_items(feats, ids)
                feats = []
                ids = []

        if len(feats) > 0:
            self.index.add_items(feats, ids)


    def load_all_edges(self):
        print("MemoryGraph: loading graph")
        start = MemoryGraph.edge_key((0,0))
        stop = MemoryGraph.edge_key((MAX_KEY_VALUE, MAX_KEY_VALUE))

        for b in self.db.iterator(start=start, stop=stop, include_value=False):
            from_node_id = struct.unpack_from('>Q', b, offset=1)[0]
            to_node_id = struct.unpack_from('>Q', b, offset=9)[0]
            cwg.add_edge(self.graph, from_node_id, to_node_id)


    #######################################################

    def generate_node_ids(self, count):
        return [self.generate_id(MemoryGraph.node_key) for _ in range(count)]

    def generate_observation_ids(self, count):
        return [self.generate_id(MemoryGraph.observation_key) for _ in range(count)]

    def generate_id(self, key_fn):
        while True:
            id = random.getrandbits(64)
            b = self.db.get(key_fn(id))
            if b is None:
                return id

    ######################################################


    ######################
    # NODES
    ######################

    @staticmethod
    def numpy_to_bytes(a):
        return a.tobytes()

    @staticmethod
    def numpy_from_bytes(b):
        return np.frombuffer(b, dtype=np.float32)


    @staticmethod
    def community_key(node_id, walk_length, walk_trials, member_portion):
        return b'y' + struct.pack('>Q', node_id) + b':' + struct.pack('>I', walk_length) + b':' + struct.pack('>I', walk_trials) + b':' + struct.pack('>I', member_portion)

    @staticmethod
    def encode_community(community):
        b = bytearray()
        for c in community:
            b.extend(struct.pack('>Q', c))
        return bytes(b)

    @staticmethod
    def decode_community(b):
        return [struct.unpack_from('>Q', b, offset=i*8)[0] for i in range(int(len(b)/8))]

    # node:[node_id] -> [node_data]
    @staticmethod
    def encode_node(node):
        b = MemoryGraph.numpy_to_bytes(node["f"])
        # if "c" in node:
        #    b +=  MemoryGraph.encode_community(node["c"])
        return b

    @staticmethod
    def decode_node_key(k):
        return struct.unpack_from('>Q', k)[0]

    @staticmethod
    def decode_node(v):
        node = dict()
        node["f"] = MemoryGraph.numpy_from_bytes(v[0:(4*512)])
        # if len(v) > 4*512:
        #     node["c"] =  MemoryGraph.decode_community(v[4*512:])
        return node

    @staticmethod
    def decode_node_key(k):
        return struct.unpack_from('>Q', k, offset=1)[0]

    @staticmethod
    def node_key(node_id):
        return b'n' + struct.pack('>Q', node_id)

    def get_node(self, node_id):
        return self.get_nodes([node_id])[0]

    def insert_node(self, node):
        return self.insert_nodes([node])[0]

    def get_nodes(self, node_ids):
        return [{"f":f} for f in self.index.get_items(node_ids)]

    # TODO: should be parallelizable safe (plyvel, hnswlib, networkx)
    def insert_nodes(self, nodes):
        node_ids = self.generate_node_ids(len(nodes))

        wb = self.db.write_batch()
        for node_id, node in zip(node_ids, nodes):
            cwg.add_node(self.graph, node_id)
            wb.put(MemoryGraph.node_key(node_id), MemoryGraph.encode_node(node))
        wb.write()

        self.index.add_items([n["f"] for n in nodes], node_ids)

        return node_ids

    def read_node(self, node_id):
        b = self.db.get(MemoryGraph.node_key(node_id))
        if b is None:
            return None
        else:
            node = MemoryGraph.decode_node(b)
            node["id"] = node_id
            return node

    def write_node(self, node):
        self.db.put(MemoryGraph.node_key(node["id"]), MemoryGraph.encode_node(node))

    def read_community(self, node_id, walk_length, walk_trials, member_portion):
        b = self.db.get(MemoryGraph.community_key(node_id, walk_length, walk_trials, member_portion))
        if b is None:
            return None
        return MemoryGraph.decode_community(b)

    def write_community(self, node_id, walk_length, walk_trials, member_portion, community):
        self.db.put(MemoryGraph.community_key(node_id, walk_length, walk_trials, member_portion), MemoryGraph.encode_community(community))

    ######################
    # EDGES
    ######################

    @staticmethod
    def edge_key(edge):
        return b'e' + struct.pack('>Q', edge[0]) + struct.pack('>Q', edge[1])

    # TODO: should be parallelizable safe (plyvel)
    def save_edges(self, edges):
        wb = self.db.write_batch()
        for edge in edges:
            wb.put(MemoryGraph.edge_key(edge), b'')
        wb.write()


    ######################
    # Counts
    ######################

    @staticmethod
    def pixel_object_count_key(obj):
        return b'c:p:' + obj.encode()

    @staticmethod
    def pixel_count_key():
        return b'c:p'

    @staticmethod
    def frame_object_count_key(obj):
        return b'c:f:' + obj.encode()

    @staticmethod
    def frame_count_key():
        return b'c:f'

    @staticmethod
    def observation_object_count_key(obj):
        return b'c:o:' + obj.encode()

    @staticmethod
    def observation_count_key():
        return b'c:o'

    @staticmethod
    def video_object_count_key(obj):
        return b'c:v:' + obj.encode()

    @staticmethod
    def video_count_key():
        return b'c:v'

    def increment_count_wb(self, wb, key, amount):
        c = self.get_count(key)
        wb.put(key, struct.pack('>Q', c + amount))

    def get_count(self, key):
        c = self.db.get(key)
        if c is None:
            return 0
        else:
            return struct.unpack_from('>Q', c)[0]

    def get_counts(self):
        observation_count = self.get_count(MemoryGraph.observation_count_key())
        observation_objects = dict()
        for k,v in self.db.iterator(start=b'c:o:', stop=b'c:o:~'):
            observation_objects[k.decode().split(':')[2]] = struct.unpack_from('>Q', v)[0]

        frame_count = self.get_count(MemoryGraph.frame_count_key())
        frame_objects = dict()
        for k,v in self.db.iterator(start=b'c:f:', stop=b'c:f:~'):
            frame_objects[k.decode().split(':')[2]] = struct.unpack_from('>Q', v)[0]

        video_count = self.get_count(MemoryGraph.video_count_key())
        video_objects = dict()
        for k,v in self.db.iterator(start=b'c:v:', stop=b'c:v:~'):
            video_objects[k.decode().split(':')[2]] = struct.unpack_from('>Q', v)[0]

        pixel_count = self.get_count(MemoryGraph.pixel_count_key())
        pixel_objects = dict()
        for k,v in self.db.iterator(start=b'c:p:', stop=b'c:p:~'):
            pixel_objects[k.decode().split(':')[2]] = struct.unpack_from('>Q', v)[0]

        return {
            "observation_count": observation_count,
            "observation_objects": observation_objects,
            "frame_count": frame_count,
            "frame_objects": frame_objects,
            "video_count": video_count,
            "video_objects": video_objects,
            "pixel_count": pixel_count,
            "pixel_objects": pixel_objects,
        }


    # objects: a set of object names
    def increment_video_counts(self, objects):
        wb = self.db.write_batch()
        for obj in objects:
            self.increment_count_wb(wb, MemoryGraph.video_object_count_key(obj), 1)
        self.increment_count_wb(wb, MemoryGraph.video_count_key(), 1)
        wb.write()

    # object_pixels: a dict of object names -> pixels in object
    def increment_frame_counts(self, pixels, object_pixels):
        wb = self.db.write_batch()
        for obj, pix in object_pixels.items():
            self.increment_count_wb(wb, MemoryGraph.frame_object_count_key(obj), 1)
            self.increment_count_wb(wb, MemoryGraph.pixel_object_count_key(obj), pix)
        self.increment_count_wb(wb, MemoryGraph.frame_count_key(), 1)
        self.increment_count_wb(wb, MemoryGraph.pixel_count_key(), pixels)
        wb.write()


    ######################
    # OBSERVATIONS
    ######################
    # {"file":file, "t":t, "y":y, "x":x, "patch":patch}

    @staticmethod
    def encode_observation(observation):
        bt = struct.pack('>I', observation["t"]) # 4 bytes
        by = struct.pack('>d', observation["y"]) # 8 bytes
        bx = struct.pack('>d', observation["x"]) # 8 bytes
        #bpatch = observation["patch"].tobytes() # 3072 bytes

        if "o" in observation and observation["o"] is not None:
            bo = observation["o"].encode()
        else:
            bo = b''

        bolen = struct.pack('>H', len(bo)) # 2 bytes
        
        bfile = observation["file"].encode()
        bfilelen = struct.pack('>H', len(bfile)) # 2 bytes
        
        return bt + by + bx + bolen + bo + bfilelen + bfile


    @staticmethod
    def decode_observation(b):
        observation = dict()
        observation["t"] = struct.unpack_from('>I', b, offset=0)[0]
        observation["y"] = struct.unpack_from('>d', b, offset=4)[0]
        observation["x"] = struct.unpack_from('>d', b, offset=12)[0]
        # observation["patch"] = np.frombuffer(b[20:3092], dtype=np.uint8).reshape(32, 32, 3)

        offset = 20
        olen = struct.unpack_from('>H', b, offset=offset)[0]
        offset += 2
        if olen > 0:
            observation["o"] = b[offset:offset+olen].decode()
        offset += olen
        filelen = struct.unpack_from('>H', b, offset=offset)[0]
        offset += 2
        observation["file"] = b[offset:offset+filelen].decode()

        return observation


    # obs:[observation_id] -> [observation_data]
    @staticmethod
    def observation_key(observation_id):
        return b'o' + struct.pack('>Q', observation_id)

    # get observation - observation is a dictionary
    def get_observation(self, observation_id):
        b = self.db.get(MemoryGraph.observation_key(observation_id))
        observation = MemoryGraph.decode_observation(b)
        observation["id"] = observation_id
        return observation

    def insert_observation(self, observation):
        return self.insert_observations([observation])[0]

    def get_observations(self, observation_ids):
        return [self.get_observation(observation_id) for observation_id in observation_ids]

    # TODO: each observation should have a list of the objects that where in the frame
    
    # TODO: should be parallelizable safe (plyvel)
    def insert_observations(self, observations):
        observation_ids = self.generate_observation_ids(len(observations))
        wb = self.db.write_batch()

        for observation_id, observation in zip(observation_ids, observations):
            b = MemoryGraph.encode_observation(observation)
            wb.put(MemoryGraph.observation_key(observation_id), b)
            if "o" in observation and observation["o"] is not None:
                self.increment_count_wb(wb, MemoryGraph.observation_object_count_key(observation["o"]), 1)

        self.increment_count_wb(wb, MemoryGraph.observation_count_key(), len(observations))
        
        wb.write()
        return observation_ids

    # integrated_observation:[node_id]:[observation_id]
    @staticmethod
    def integrated_observations_key(node_id, observation_id):
        return b'i' + struct.pack('>Q', node_id) + struct.pack('>Q', observation_id)

    # observations that are integrated into node's features
    def get_integrated_observations(self, node_id):
        start = MemoryGraph.integrated_observations_key(node_id, 0)
        stop = MemoryGraph.integrated_observations_key(node_id, MAX_KEY_VALUE)
        return [struct.unpack_from('>Q', b, offset=9)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]

    # TODO: should be parallelizable safe (plyvel)
    def add_integrated_observations(self, node_ids, observation_ids):
        wb = self.db.write_batch()
        for node_id, observation_id in zip(node_ids, observation_ids):
            wb.put(MemoryGraph.integrated_observations_key(node_id, observation_id), b'')
            wb.put(MemoryGraph.integrated_nodes_key(observation_id, node_id), b'')
        wb.write()


    # predicted_observation:[node_id]:[observation_id]
    @staticmethod
    def predicted_observations_key(node_id, observation_id):
        return b'p' + struct.pack('>Q', node_id) + struct.pack('>Q', observation_id)

    # observations that were predicted by node
    def get_predicted_observations(self, node_id):
        start = MemoryGraph.predicted_observations_key(node_id, 0)
        stop = MemoryGraph.predicted_observations_key(node_id, MAX_KEY_VALUE)
        return [struct.unpack_from('>Q', b, offset=9)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]

    # TODO: should be parallelizable safe (plyvel)
    def add_predicted_observations(self, node_ids, observation_ids):
        wb = self.db.write_batch()
        for node_id, observation_id in zip(node_ids, observation_ids):
            wb.put(MemoryGraph.predicted_observations_key(node_id, observation_id), b'')
            wb.put(MemoryGraph.predicted_nodes_key(observation_id, node_id), b'')
        wb.write()


    # predicted_node:[observation_id]:[node_id]
    @staticmethod
    def predicted_nodes_key(observation_id, node_id):
        return b'q' + struct.pack('>Q', observation_id) + struct.pack('>Q', node_id)
    
    # nodes that predicted observation
    def get_predicted_nodes(self, observation_id):
        start = MemoryGraph.predicted_nodes_key(observation_id, 0)
        stop = MemoryGraph.predicted_nodes_key(observation_id, MAX_KEY_VALUE)
        return [struct.unpack_from('>Q', b, offset=9)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]


    # integrated_node:[observation_id]:[node_id]
    @staticmethod
    def integrated_nodes_key(observation_id, node_id):
        return b'j' + struct.pack('>Q', observation_id) + struct.pack('>Q', node_id)

    # nodes that integrate observation
    def get_integrated_nodes(self, observation_id):
        start = MemoryGraph.integrated_nodes_key(observation_id, 0)
        stop = MemoryGraph.integrated_nodes_key(observation_id, MAX_KEY_VALUE)
        return [struct.unpack_from('>Q', b, offset=9)[0] for b in self.db.iterator(start=start, stop=stop, include_value=False)]


    def get_adjacencies(self, node_id, radius):
        cwg.neighbors(self.graph, node_id, radius)
   
    def insert_adjacency(self, from_id, to_id):
        if(from_id == to_id): return
        self.save_edges([(from_id, to_id)])
        cwg.add_edge(self.graph, from_id, to_id)


    def insert_adjacencies(self, edges):
        edges = [e for e in edges if e[0] != e[1]]
        self.save_edges(edges)
        for e in edges:
            cwg.add_edge(self.graph, e[0], e[1])


    def knn_query(self, feats, k=1):
        if len(feats) == 0:
            return ([],[])
        return self.index.knn_query(feats, k)   

    def index_count(self):
        return cwg.len(self.graph)

    
    def get_communities(self, node_ids, walk_length=10, walk_trials=1000, member_portion=200):
        return cwg.communities(self.graph, node_ids, walk_length, walk_trials, member_portion)

    def get_communities_range(self, node_ids, log2_min_len=3, log2_max_len=12, walk_trials=1000, member_portion=200):
        return cwg.communities_range(self.graph, node_ids, log2_min_len, log2_max_len, walk_trials, member_portion)

    def get_community(self, node_id, walk_length=10, walk_trials=1000, member_portion=200, save_to_db=True):
        
        if save_to_db:
            community = self.read_community(node_id, walk_length, walk_trials, member_portion)
            if community is not None:
                return set(community)
  
        community = cwg.community(self.graph, node_id, walk_length, walk_trials, member_portion)

        if save_to_db:
            self.write_community(node_id, walk_length, walk_trials, member_portion, community)

        return set(community)



    def observations_for_nodes(self, node_ids):
        observation_ids = []
        for node_id in node_ids:
            integrated_observations = self.get_integrated_observations(node_id)
            observation_ids.extend(integrated_observations)
            predicted_observations = self.get_predicted_observations(node_id)
            observation_ids.extend(predicted_observations)
        return observation_ids


    # searches using the max pool of the provided feature cluster
    def search_group_foo(self, features, params):
        if len(features) == 0:
            return set()

        features_max = np.max(features, axis=0)
        lab, dis = self.knn_query([features_max], k=params["search_knn"])

        labels_merged = list(chain.from_iterable(lab))
        distances_merged = list(chain.from_iterable(dis))

        neighbor_nodes_merged = list(set([l for l,d in zip(labels_merged, distances_merged) if d <= params["feature_dis"]]))

        results = set()

        communities = self.get_communities(neighbor_nodes_merged, walk_length= params["initial_walk_length"], walk_trials=params["walk_trials"], member_portion=params["member_portion"])

        for i in range(len(neighbor_nodes_merged)):
            
            walk_length =  params["initial_walk_length"]
            last_community = frozenset()

            while True: # 16 32 64 128 256 512 1024 2048 4096
                if walk_length >= params["max_walk_length"]:
                    break

                if walk_length ==  params["initial_walk_length"]:
                    community = frozenset(communities[i])
                else:
                    community = frozenset(self.get_communities([neighbor_nodes_merged[i]], walk_length=walk_length, walk_trials=params["walk_trials"], member_portion=params["member_portion"])[0])

                if last_community == community:
                    break

                community_features_list = [i for i in [self.get_node(c)["f"] for c in community] if i is not None]
                if len(community_features_list) == 0:
                    break
                community_features = np.array(community_features_list)
                community_features_max = np.max(community_features, axis=0)
                d = self.distance(community_features_max, features_max)
                print(walk_length, d, len(community))

                if d > params["community_dis"]:
                    break

                last_community = community
                walk_length = walk_length * 2 

            results.add(last_community)

        return results


    # this should find nearest groups and return them in the order of nearness
    def search_group(self, features, params):
        
        if len(features) == 0:
            return set()

        lab, dis = self.knn_query(features, k=params["search_knn"])
        features_max = np.max(features, axis=0)
        
        labels_merged = list(chain.from_iterable(lab))
        distances_merged = list(chain.from_iterable(dis))

        neighbor_nodes_merged = list(set([l for l,d in zip(labels_merged, distances_merged) if d <= params["feature_dis"]]))

        results = set()

        communities = self.get_communities(neighbor_nodes_merged, walk_length= params["initial_walk_length"], walk_trials=params["walk_trials"], member_portion=params["member_portion"])

        for i in range(len(neighbor_nodes_merged)):
            
            walk_length =  params["initial_walk_length"]
            last_community = frozenset()

            while True: # 16 32 64 128 256 512 1024 2048 4096
                if walk_length >= params["max_walk_length"]:
                    break
                
                if walk_length ==  params["initial_walk_length"]:
                    community = frozenset(communities[i])
                else:
                    community = frozenset(self.get_communities([neighbor_nodes_merged[i]], walk_length=walk_length, walk_trials=params["walk_trials"], member_portion=params["member_portion"])[0])

                if last_community == community:
                    break

                
                community_features_list = [i for i in [self.get_node(c)["f"] for c in community] if i is not None]
                if len(community_features_list) == 0:
                    break
                community_features = np.array(community_features_list)
                community_features_max = np.max(community_features, axis=0)
                d = self.distance(community_features_max, features_max)
                # print(walk_length, d, len(community))

                if d >  params["community_dis"]:
                    break

                last_community = community
                walk_length = walk_length * 2 

            results.add(last_community)

        return results


    def distance(self, a, b):
        if self.space == 'cosine':
            return np_cosine(a, b)
        else:
            return np.linalg.norm(a-b)


def np_cosine(x,y):
    return 1 - np.inner(x,y)/math.sqrt(np.dot(x,x)*np.dot(y,y))


def get_rad_grid(g_pos, rad, shape, stride):

    top_left = (g_pos[0]-rad, g_pos[1]-rad)
    g_width = math.floor((shape[0] - 32)/stride)
    g_height = math.floor((shape[1] - 32)/stride)

    res = []

    for i in range(2*rad+1):
        p = (top_left[0]+i, top_left[1])
        if p[0] >= 0 and p[1] >= 0 and p[0] < g_width and p[1] < g_height:
            res.append(p)
 
    for i in range(2*rad+1):
        p = (top_left[0]+i, top_left[1]+(2*rad+1))
        if p[0] >= 0 and p[1] >= 0 and p[0] < g_width and p[1] < g_height:
            res.append(p)

    for i in range(2*rad-1):
        p = (top_left[0], top_left[1]+(i+1))
        if p[0] >= 0 and p[1] >= 0 and p[0] < g_width and p[1] < g_height:
            res.append(p)

    for i in range(2*rad-1):
        p = (top_left[0]+(2*rad), top_left[1]+(i+1))
        if p[0] >= 0 and p[1] >= 0 and p[0] < g_width and p[1] < g_height:
            res.append(p)

    #print(rad, g_pos, res)
    return res



def first_pos(kp_grid):
    ## TODO: if there are no key points in frame
    loc = random.choice(list(kp_grid.keys()))
    return loc, random.choice(kp_grid[loc])


def next_pos(kp_grid, shape, g_pos, walk_length, stride):
 
    if (g_pos is not None) and (random.random() > 1.0/walk_length):

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


# def object_names_from_image_file(image_file):
#     return re.findall('_([a-z]+)', image_file)


# def pixel_counts(obj_frame, center_size):
#     unique, counts = np.unique(obj_frame, return_counts=True)
#     unique = [object_name_for_idx(o) for o in unique]

#     min_pix = center_size * center_size * 0.9
#     pixels = dict([(u, c) for u, c in zip(unique, counts) if u is not None and c > min_pix])
    
#     return pixels


# def extract_object(window, center_size):
#     c = np.bincount(window.flatten())
#     if np.max(c) >= center_size*center_size*.90:
#         return object_name_for_idx(np.argmax(c))
#     else:
#         return None


# def extract_objects(obj_frame, pos, center_size):
#     windows = np.empty((len(pos), center_size, center_size), dtype=np.uint8)

#     for i in range(len(pos)):
#         windows[i] = extract_window(obj_frame, pos[i], center_size)

#     return [extract_object(w, center_size) for w in windows]


def extract_windows(frame, pos, window_size):
    windows = np.empty((len(pos), window_size, window_size, 3), dtype=np.uint8)

    for i in range(len(pos)):
        windows[i] = extract_window(frame, pos[i], window_size)

    return windows



def extract_window(frame, pos, window_size):
    half_w = window_size/2.0
    bottom_left = [int(round(pos[0]-half_w)), int(round(pos[1]-half_w))]
    top_right = [bottom_left[0]+window_size, bottom_left[1]+window_size]
   
    if bottom_left[0] < 0:
        top_right[0] -= bottom_left[0]
        bottom_left[0] = 0

    if bottom_left[1] < 0:
        top_right[1] -= bottom_left[1]
        bottom_left[1] = 0

    if top_right[0] >= frame.shape[0]:
        bottom_left[0] -= (top_right[0]-frame.shape[0]+1)
        top_right[0] = frame.shape[0]-1

    if top_right[1] >= frame.shape[1]:
        bottom_left[1] -= (top_right[1]-frame.shape[1]+1)
        top_right[1] = frame.shape[1]-1

    return frame[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]


def key_point_grid(orb, frame, stride):

    kp = orb.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)

    grid = dict()

    grid_offset_x = ((frame.shape[0] - 32) % stride)/2.0 + 16
    grid_offset_y = ((frame.shape[1] - 32) % stride)/2.0 + 16

    for k in kp:
        p = (k.pt[1],k.pt[0])
        g = (int(math.floor((p[0]-grid_offset_x)/stride)), int(math.floor((p[1]-grid_offset_y)/stride)))
        if g in grid:
            grid[g].append(p)
        else:
            grid[g] = [p]

    return grid

# def extract_window_pixels(pos, frame_shape, window_size):
#     half_w = window_size/2.0
#     bottom_left = [int(round(pos[0]-half_w)), int(round(pos[1]-half_w))]
#     top_right = [bottom_left[0]+window_size, bottom_left[1]+window_size]
   
#     if bottom_left[0] < 0:
#         top_right[0] -= bottom_left[0]
#         bottom_left[0] = 0

#     if bottom_left[1] < 0:
#         top_right[1] -= bottom_left[1]
#         bottom_left[1] = 0

#     if top_right[0] >= frame_shape[0]:
#         bottom_left[0] -= (top_right[0]-frame_shape[0]+1)
#         top_right[0] = frame_shape[0]-1

#     if top_right[1] >= frame_shape[1]:
#         bottom_left[1] -= (top_right[1]-frame_shape[1]+1)
#         top_right[1] = frame_shape[1]-1

#     points = []
#     for y in range(bottom_left[0], top_right[0]):
#         for x in range(bottom_left[1], top_right[1]):
#             points.append((y,x))
            
#     return points
    



###############
# Build Graph #
###############

def build_graph(db_path, image_files, params):

    print("Starting...")

    random.shuffle(image_files)

    orb = cv2.ORB_create(nfeatures=100000, fastThreshold=7)

    # initialize VGG16
    model = ShufflePatchFeatureExtractor("/Users/racoon/Desktop/variation_2b_migrated_0135_0.001_1.4328_63.80.pt")
    #model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3)) ### TODO: build interface so underlying model can be changed

    # memory graph
    memory_graph = MemoryGraph(db_path, params)
    memory_graph_walker = MemoryGraphWalker(memory_graph, params)
    
    # for each run though the video
    for r in range(params["runs"]):

        print("Run", r)

        image_file_count = 0

        for image_file in image_files:
            
            print(image_file)

            image_file_count += 1

            pil_image = Image.open(image_file).convert('RGB')
            image = np.array(pil_image)

            g_pos = [None for _ in range(params["walker_count"])]
            pos = [None for _ in range(params["walker_count"])]
            adj = [False for _ in range(params["walker_count"])]

            done = False

            # for each frame
            for t in range(params["max_frames"]):
                if done:
                    break

                kp_grid = key_point_grid(orb, image, params["stride"])

                for i in range(params["walker_count"]):
                    g_pos[i], pos[i], adj[i] = next_pos(kp_grid, image.shape, g_pos[i], params["walk_length"], params["stride"])

                patches = extract_windows(image, pos, params["window_size"])
                windows = patches.astype(np.float64)

                # extract cnn features from windows
                # preprocess_input(windows) ### TODO: these methods should be part of the model interface from above
                # feats = model.predict(windows)

                feats = model.evalRGB(windows)
                feats = feats.reshape((windows.shape[0], 512))
  
                # objects = extract_objects(obj_frame, pos, params["center_size"])
                ids = memory_graph_walker.add_parrelell_observations(image_file, t, pos, adj, feats, patches)


                # STATS

                restart_count = 0
                near_neighbor_count = 0
                is_identical_count = 0
                has_predictions_count = 0
                has_accurate_predictions_count = 0
                has_too_many_accurate_predictions_count = 0
                adjacencies_inserted = 0
                prediction_count = 0
                prediction_members_count = 0
                nn_gte_15 = 0
                nn_gte_30 = 0
                nn_gte_50 = 0
                skipped = 0
                adjacencies_skipped = 0

                time_stats = dict()

                for i in range(params["walker_count"]):
                    if ids[i][0] is None:
                        # restart walk because we are in a very predictable spot
                        g_pos[i] = None
                        pos[i] = None
                        adj[i] = False  

                    stats = ids[i][2]

                    if not stats["adj"]:
                        restart_count += 1
                    if "nearest_neighbor" in stats and stats["nearest_neighbor"] < params["distance_threshold"]:
                        near_neighbor_count += 1
                    if "skipped" in stats and stats["skipped"]:
                        skipped += 1
                    if "adjacencies_skipped" in stats :
                        adjacencies_skipped += stats["adjacencies_skipped"]
                    if "predictions" in stats:
                        has_predictions_count += 1
                        prediction_count += stats["predictions"]
                    if "prediction_members" in stats:
                        prediction_members_count += stats["prediction_members"]
                    if "accurate_predictions" in stats:
                        if stats["accurate_predictions"] > 0:
                            has_accurate_predictions_count += 1
                        if stats["accurate_predictions"] >= memory_graph_walker.accurate_prediction_limit:
                            has_too_many_accurate_predictions_count += 1
                    if "identical" in stats and stats["identical"]:
                        is_identical_count += 1
                    if "adjacencies_inserted" in stats:
                        adjacencies_inserted += stats["adjacencies_inserted"]
                    if "near_neighbors_count" in stats:
                        if stats["near_neighbors_count"] >= 15:
                            nn_gte_15 += 1
                        if stats["near_neighbors_count"] >= 30:
                            nn_gte_30 += 1
                        if stats["near_neighbors_count"] >= 50:
                            nn_gte_50 += 1
                    if params["keep_times"]:
                        for k, v in stats["time"].items():
                            if k not in time_stats:
                                time_stats[k] = v
                            else:
                                time_stats[k] += v

                if params["keep_times"]:
                    print(time_stats)

                print(
                    "img", image_file_count, 
                    "frame", t+1,
                    "start", restart_count, 
                    "nn00", near_neighbor_count,
                    "nn15", nn_gte_15,
                    "nn30", nn_gte_30,
                    "nn50", nn_gte_50,
                    "iden", is_identical_count,
                    # "pred", has_predictions_count,
                    "accu", has_accurate_predictions_count,
                    "many", has_too_many_accurate_predictions_count,
                    # "obj", observations_with_objects,
                    "adj", adjacencies_inserted,
                    # "skp", skipped,
                    # "askp", adjacencies_skipped
                )

    counts = memory_graph.get_counts()
    print("counts", counts)

    memory_graph.close()

    print("Done")



# utility for working with marks and intervals
# a mark is a nanosecond timestamp returned from time.time_ns()
# an interval is time elapsed between two marks
class TimeMarker:
    def __init__(self, enabled=True, l="start"):
        self.enabled = enabled
        if enabled:
            self.last = time.time_ns()
            self.mark_dict = {l: self.last}
            self.saved = {}

    # sets a new time mark and calculates the interval from the new time mark to a previous time mark
    # l: give this mark a label to be used later
    # i: calculate the interval since a labeled mark instead of using simply the last mark
    # s: save the interval in a dict with this name
    # a: add this interval to the value in saved dict 
    # p: print the interval with the given text
    # si: a shortcut to set s and i to the same value
    def mark(self, l=None, i=None, s=None, a=None, p=None, si=None):
        if not self.enabled:
            return 0

        if si is not None:
            s = si
            i = si

        t = time.time_ns()
        
        if l is not None:
            self.mark_dict[l] = t
        
        if i is not None:
            r = t - self.mark_dict[i]     
        else:
            r = t - self.last

        self.last = t

        if s is not None:
            self.saved[s] = r
        elif a is not None:
            if a not in self.saved:
                self.saved[a] = r
            else:
                self.saved[a] += r
        if p is not None:
            print(p, r)

        return r


PARAMETERS = {
	"runs": 1,
    "window_size": 32, 
	"grid_margin": 16, 
    "max_frames": 30*5,
	"search_max_frames": 30, 
	"max_elements": 12000000,
    "space": 'cosine', 
    "dim": 512, 
    "ef": 300, 
    "M": 64, 
	"rebuild_index": False,
    "keep_times": False,

	"stride": 24,
	"center_size": 16,
	"walk_length": 100,
    "walker_count": 200,
    "prevent_similar_adjacencies": False,
    "knn": 50,
    "accurate_prediction_limit": 12,
    "distance_threshold": 0.15,
    "prediction_history_length": 7,
    "history_community_matches": 1,
    "identical_distance": 0.15,
    "search_walker_count": 4,
    "search_walk_length": 10,
    "feature_dis": 0.4,
    "community_dis": 0.15,
    "search_knn": 100,
    "initial_walk_length": 8,  
    "max_walk_length": 4096,
    "member_portion": 100,
    "walk_trials": 1000,
}