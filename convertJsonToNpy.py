import argparse 
import numpy as np
import json

parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--json_folder', type=str, required=True, help="folder path of json files")




def convert_json_to_npy(json_file):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    frames = data['frames'] # frames
    T = len(frames) # number of frames 
    V = data['njts'] # number of joints
    C = 3 # 3D coordinates (x, y, z)
    M = 1 #number of Skeletons

    tensor = np.zeros((T, V, C, M), dtype=np.float32) # Initialize the tensor
    for t, frame in enumerate(frames):
        if not frame: # Check if the frame is empty
            continue
        person = frame[0] # Get the first person in the frame
        pose3d = person['pose3d']
        joints = np.array(pose3d).reshape(V, C)


        for c in range(C):
            tensor[c, t, :, 0] = frames[:][c] # Fill the tensor with the joint coordinates
        

        return tensor
