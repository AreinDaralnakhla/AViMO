import sys
import pickle
import numpy as np

from premiere.functionsCommon import buildTracksForFusion, computeMaxId

def analyzeTrack(tracks, trackId):    
    # Extraire les idSAM pour le trackId donné
    idSAMs = tracks[trackId][:, 2]    
    # Filtrer les valeurs -1
    idSAMs = idSAMs[idSAMs != -1] 
    if len(idSAMs) == 0:
        return -1
    # Compter les occurrences de chaque valeur dans idSAMs
    counts = np.bincount(idSAMs) 
    # Trouver la valeur la plus fréquente
    most_frequent_idSAM = np.argmax(counts)
    return most_frequent_idSAM

if len(sys.argv) < 4:
    print("Usage: python tracksFusion.py <input_pkl_path> <output_pkl_path> <trackSizeMin>")
    sys.exit(1)

# Define pathing & device usage
pkl_path = sys.argv[1]

# Open the pkl file
print ("read pkl: ", pkl_path)
file = open(pkl_path, 'rb')
dataPKL = pickle.load(file) 
file.close()

trackSizeMin = int(sys.argv[3])
allFrameHumans = dataPKL['allFrameHumans']

maxId = computeMaxId(allFrameHumans)
tracks, tracksSize = buildTracksForFusion(allFrameHumans, maxId)

for i in range(len(tracksSize)):
    most_frequent = analyzeTrack ( tracks, i)
#    print (f'Track {i}: {tracksSize[i]} - {most_frequent}')
    if (most_frequent != -1) and (tracksSize[i] > trackSizeMin):
        for j in range(tracksSize[i]):
            allFrameHumans[tracks[i][j][0]][tracks[i][j][1]]['id'] = most_frequent
    else:
       for j in range(tracksSize[i]):
            allFrameHumans[tracks[i][j][0]][tracks[i][j][1]]['id'] = -1

for i in range(len(allFrameHumans)):
    unique_humans = []
    seen_ids = set()
    for human in allFrameHumans[i]:
        if human['id'] not in seen_ids:
            seen_ids.add(human['id'])
            unique_humans.append(human)
    allFrameHumans[i] = unique_humans
              
with open(sys.argv[2], 'wb') as handle:
    pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL) 

# python .\tracksFusion.py .\pkl\D0-21-896L-clean-track-seg.pkl .\pkl\D0-21-896L-clean-track-seg-fus.pkl 2
