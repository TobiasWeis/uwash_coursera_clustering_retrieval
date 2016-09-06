#!/usr/bin/python

import sframe                                                  # see below for install instruction
import matplotlib.pyplot as plt                                # plotting
import numpy as np                                             # dense matrices
from scipy.sparse import csr_matrix                            # sparse matrices
from sklearn.preprocessing import normalize                    # normalizing vectors
from sklearn.metrics import pairwise_distances                 # pairwise distances
import sys      
import os

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix( (data, indices, indptr), shape)

wiki = sframe.SFrame('data/people_wiki.gl/')
tf_idf = load_sparse_csr('data/people_wiki_tf_idf.npz')
map_index_to_word = sframe.SFrame('data/people_wiki_map_index_to_word.gl/')

tf_idf = normalize(tf_idf)

def get_initial_centroids(data, k, seed=None):
    '''Randomly choose k data points as initial centroids'''
    if seed is not None: # useful for obtaining consistent results
        np.random.seed(seed)
    n = data.shape[0] # number of data points
        
    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)
    
    # Keep centroids as dense format, as many entries will be nonzero due to averaging.
    # As long as at least one document in a cluster contains a word,
    # it will carry a nonzero weight in the TF-IDF vector of the centroid.
    centroids = data[rand_indices,:].toarray()
    
    return centroids


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix( (data, indices, indptr), shape)

from sklearn.metrics.pairwise import pairwise_distances

# Get the TF-IDF vectors for documents 100 through 102.
queries = tf_idf[100:102,:]

# Compute pairwise distances from every data point to each query vector.
dist = pairwise_distances(tf_idf, queries, metric='euclidean')

print dist
# dist[i,j] -> dist between ith row of X (X[i,:]) and jth row of Y (Y[j,:])

# Assume first three docs are cluster centers
# compute distance of all docs to cluster centers
# save the distance of doc 430 to cluster center #1 (2nd one), run checkpoint
distances = pairwise_distances(tf_idf, tf_idf[:3])
dist = distances[430,1]
############################
# Checkpoint
############################
if np.allclose(dist, pairwise_distances(tf_idf[430,:], tf_idf[1,:])):
    print('Pass Checkpoint 1')
else:
    print('Check your code again (Checkpoint 1)')

# Read doc of np.argmin, write code to produce 1D array whose i-th entry indicates the closest centorid to i-th data point
closest_cluster = np.argmin(distances, axis=1)

############################
# Checkpoint
############################
reference = [list(row).index(min(row)) for row in distances]
if np.allclose(closest_cluster, reference):
    print('Pass Checkpoint 2')
else:
    print('Check your code again (Checkpoint 2)')

############################
# Checkpoint
############################
def assign_clusters(data, centroids):
    
    # Compute distances between each data point and the set of centroids:
    # Fill in the blank (RHS only)
    distances_from_centroids = pairwise_distances(data,centroids)
    
    # Compute cluster assignments for each data point:
    # Fill in the blank (RHS only)
    cluster_assignment = np.argmin(distances_from_centroids, axis=1)
    
    return cluster_assignment


cluster_assignment = assign_clusters(tf_idf, tf_idf[:3])

if len(cluster_assignment)==59071 and \
           np.array_equal(np.bincount(cluster_assignment), np.array([23061, 10086, 25924])):
   print('Pass Checkpoint 3') # count number of data points for each cluster
else:
   print('Check your code again. (Checkpoint 3)')

############################
# Checkpoint
############################
if np.allclose(assign_clusters(tf_idf[0:100:10], tf_idf[0:8:2]), np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1])):
    print('Pass Checkpoint 4')
else:
    print('Check your code again. (Checkpoint 4)')

# Resume at: "Revising Clusters"
