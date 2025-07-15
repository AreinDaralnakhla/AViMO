"""
Clean and cluster humans in frames from a pickled dataset.

This script reads a pickle file containing a dictionary with 
key 'allFrameHumans', where each frame stores a list of detected humans.
Depending on the 'nbMaxHumans' parameter:
  - If nbMaxHumans == 1, only the highest-scoring human is kept per frame.
  - Otherwise, hierarchical clustering is applied to group humans 
    by their 3D pelvis translation. Then the top-scoring human per cluster 
    is kept, optionally limiting the total number of clusters to 'nbMaxHumans'
    if nbMaxHumans != -1.

Usage:
  python cleanFramesPkl.py <input_pkl> <output_pkl> <nbMaxHumans> <distance_threshold>

:param input_pkl: Path to the input pickle file.
:param output_pkl: Path for saving the cleaned pickle file.
:param nbMaxHumans: Maximum number of humans to keep per frame. 
                   If -1, keep all clusters.
:param distance_threshold: Distance threshold used by hierarchical clustering 
                          (ward linkage).
"""

import os
import sys
import cv2
import pickle
import copy

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Command line argument validation
if len(sys.argv) != 5:
    print("Usage: python cleanFramesPkl.py <input_pkl> <output_pkl> <nbMaxHumans> <distance_threshold>")
    sys.exit(1)

# Read arguments
input_pkl = sys.argv[1]
output_pkl = sys.argv[2]
nbMaxHumans = int(sys.argv[3])
distance_threshold = float(sys.argv[4])

print("read pkl:", input_pkl)
with open(input_pkl, 'rb') as file:
    dataPKL = pickle.load(file)

# Deep copy to avoid modifying the original data
newDataPKL = copy.deepcopy(dataPKL)

# Preallocate arrays for clusters per frame and humans per frame
NbClustersPerFrame = np.zeros(len(dataPKL['allFrameHumans']), dtype=int)
humansPerFrames = np.empty(len(dataPKL['allFrameHumans']), dtype=int)

# Count how many humans are initially in each frame
for i in range(len(dataPKL['allFrameHumans'])):
    humansPerFrames[i] = len(dataPKL['allFrameHumans'][i])

maxNbHumansIn = max(humansPerFrames)
minNbHumansIn = min(humansPerFrames)

print('maxNbHumans (before cleanup):', maxNbHumansIn)
print('minNbHumans (before cleanup):', minNbHumansIn)

# Case 1: If nbMaxHumans == 1, pick only the highest-scoring human per frame
if nbMaxHumans == 1:
    for i in range(len(dataPKL['allFrameHumans'])):
        humans = dataPKL['allFrameHumans'][i]
        # Clear newDataPKL humans for frame i
        newDataPKL['allFrameHumans'][i] = []

        if len(humans) == 0:
            # No humans in this frame
            continue
        
        # Pick the single best (highest 'score')
        scores = np.array([h['score'] for h in humans])
        maxIndex = np.argmax(scores)
        newDataPKL['allFrameHumans'][i].append(humans[maxIndex])
        
        # Only 1 cluster effectively
        NbClustersPerFrame[i] = 1

else:
    # Case 2: nbMaxHumans != 1 (could be > 1, or == -1 for unlimited)
    for i in range(len(dataPKL['allFrameHumans'])):
        humans = dataPKL['allFrameHumans'][i]
        # Clear the newDataPKL humans for frame i
        newDataPKL['allFrameHumans'][i] = []
        NbClustersPerFrame[i] = 0

        # If there are 0 or 1 humans, no need for clustering
        if len(humans) == 0:
            continue
        if len(humans) == 1:
            newDataPKL['allFrameHumans'][i].append(humans[0])
            NbClustersPerFrame[i] = 1
            continue

        # Prepare points array for hierarchical clustering (pelvis translation)
        points = np.empty([len(humans), 3], dtype=float)
        for j in range(len(humans)):
            # Extract the 3D pelvis translation for each human
            points[j, 0] = humans[j]['transl_pelvis'][0][0]
            points[j, 1] = humans[j]['transl_pelvis'][0][1]
            points[j, 2] = humans[j]['transl_pelvis'][0][2]

        # Perform hierarchical clustering with 'ward'
        Z = linkage(points, method='ward')
        clusters_by_distance = fcluster(Z, distance_threshold, criterion='distance')
        
        # Count total clusters
        num_clusters = len(set(clusters_by_distance))

        # Gather the best-scoring human from each cluster
        cluster_best_list = []
        clusters_by_id = {}
        for index, cluster_id in enumerate(clusters_by_distance):
            if (cluster_id - 1) not in clusters_by_id:
                clusters_by_id[cluster_id - 1] = []
            clusters_by_id[cluster_id - 1].append(index)
        
        # Find the best human per cluster
        for cluster_id, indices in clusters_by_id.items():
            if len(indices) > 1:
                # More than one human in this cluster
                scores = np.array([humans[idx]['score'] for idx in indices])
                maxIndex = np.argmax(scores)
                best_score = scores[maxIndex]
                best_human = humans[indices[maxIndex]]
            else:
                # Only one human in this cluster
                best_human = humans[indices[0]]
                best_score = best_human['score']
            
            cluster_best_list.append((best_score, best_human, cluster_id))

        # Sort clusters by best score in descending order
        cluster_best_list.sort(key=lambda x: x[0], reverse=True)
        
        # If nbMaxHumans == -1, do not limit number of humans
        if nbMaxHumans == -1:
            final_cluster_best_list = cluster_best_list
        else:
            # Otherwise, keep only the top nbMaxHumans clusters
            if len(cluster_best_list) > nbMaxHumans:
                final_cluster_best_list = cluster_best_list[:nbMaxHumans]
            else:
                final_cluster_best_list = cluster_best_list

        newDataPKL['allFrameHumans'][i] = []
        # Add the best humans from chosen clusters to the new data
        for (best_score, best_human, cluster_id) in final_cluster_best_list:
            newDataPKL['allFrameHumans'][i].append(best_human)

        # The final number of clusters retained
        NbClustersPerFrame[i] = len(final_cluster_best_list)

# Print cluster stats
max_clusters = max(NbClustersPerFrame)
min_clusters = min(NbClustersPerFrame)
print('max_clusters:', max_clusters)
print('min_clusters:', min_clusters)

# Recompute the number of humans in each frame after cleanup
for i in range(len(newDataPKL['allFrameHumans'])):
    humansPerFrames[i] = len(newDataPKL['allFrameHumans'][i])

maxNbHumansOut = max(humansPerFrames)
minNbHumansOut = min(humansPerFrames)

print('maxNbHumans (after cleanup):', maxNbHumansOut)
print('maxNbHumansIndices:', np.argmax(humansPerFrames))
print('minNbHumans (after cleanup):', minNbHumansOut)

# Finally, save the cleaned PKL file
with open(output_pkl, 'wb') as handle:
    pickle.dump(newDataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL)
