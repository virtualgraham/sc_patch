clusters = pickle.load(open("clusters_288_144_100.pkl", "rb"))

npz_files = glob(f'/Users/racoon/Desktop/feat_grids_{window_size_outer}_{window_size_inner}_{stride}_{version}/*/*.npz')

sub_clusters = [KMeans(n_clusters=10) for _ in range(100)]

for c in range(100):
    Y = []
    
    for idx, npz_file in tqdm(enumerate(npz_files), total=len(npz_files)):
        
        loaded = np.load(npz_file)
        feat_grid_outer = loaded['outer']
        feat_grid_inner = loaded['inner']

        grid_cluster_ids = clusters.predict(feat_grid_outer.reshape(feat_grid_outer.shape[0]*feat_grid_outer.shape[1], feat_grid_outer.shape[2]))
        grid_cluster_ids = grid_cluster_ids.reshape((feat_grid_outer.shape[0], feat_grid_outer.shape[1]))

        for y in range(feat_grid_outer.shape[0]): 
            for x in range(feat_grid_outer.shape[1]): 
                if grid_cluster_ids[y,x] == c:
                    Y.append(feat_grid_inner[y,x])
    
    print('len(Y)', len(Y))
    
    sub_clusters[c].fit(np.array(Y, dtype=np.float32))

pickle.dump(sub_clusters, open("sub_clusers.pkl", "wb"))