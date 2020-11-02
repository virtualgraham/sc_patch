import numpy as np

permutations = np.load("src/nine_patch/permutations_1000.npy")
print(permutations.shape)
print(permutations[0:9,:])