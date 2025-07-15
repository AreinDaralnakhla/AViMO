import pickle
import json
import sys

if len(sys.argv) != 4:
    print("Usage: python injectCameraPropertiesPkl.py <final_json> <input_pkl> <output_pkl>")
    sys.exit(1)

finalJSONName = sys.argv[1]
inputPklName = sys.argv[2]
outputPklName = sys.argv[3]

print ("Reading PKL: ", inputPklName)
with open(inputPklName, 'rb') as file:
    dataPKL = pickle.load(file)

print ("Reading JSON: ", finalJSONName)
with open(finalJSONName, 'r') as file:
    finalJSON = json.load(file)

dataPKL['floor_angle_deg'] = finalJSON['floor_angle_deg']
dataPKL['floor_Zoffset'] = finalJSON['floor_Zoffset']

print ("Writing PKL: ", outputPklName)
with open(outputPklName, 'wb') as file:
    pickle.dump(dataPKL, file)
