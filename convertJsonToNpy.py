import argparse 
import numpy as np
import json
import os 

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

def main():
    args = parser.parse_args()
    json_folder = args.json_folder
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    for json_file in json_files:
        json_file
        json_path = os.path.join(json_folder, json_file)
        tensor = convert_json_to_npy(json_path)
        if tensor is None:
            print(f"Skipping {json_file} due to empty frame.")
            continue
        featureTensor = tensor

        npy_file = os.path.splitext(json_file)[0] + '.npy'
        np.save(npy_file, tensor)


if __name__ == "__main__":
    main()