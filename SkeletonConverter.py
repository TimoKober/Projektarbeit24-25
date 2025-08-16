
#SmartHome skeleton topology

# 0 right ankle
# 1 left ankle
# 2 right knee
# 3 left knee
# 4 right hip
# 5 left hip
# 6 right wrist
# 7 left wrist
# 8 right elbow
# 9 left elbow
# 10 right shoulder
# 11 left shoulder
# 12 head

# NTU skeleton topology
# 0 base of spine
# 1 middle of spine
# 2 neck
# 3 head
# 4 left shoulder
# 5 left elbow
# 6 left wrist
# 7 left hand
# 8 right shoulder
# 9 right elbow 
# 10 right wrist
# 11 right hand
# 12 left hip
# 13 left knee
# 14 left ankle
# 15 left foot
# 16 right hip
# 17 right knee
# 18 right ankle
# 19 right foot
# 20 spine
# 21 tip of left hand 
# 22 left thumb
# 23 tip of right hand
# 24 right thumb

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json

shJoints = ['right ankle', 'left ankle', 'right knee', 'left knee',
            'right hip', 'left hip', 'right wrist', 'left wrist',
            'right elbow', 'left elbow', 'right shoulder', 'left shoulder', 'head']

ntuJoints = ['base of spine', 'middle of spine', 'neck', 'head',
             'left shoulder', 'left elbow', 'left wrist', 'left hand',
             'right shoulder', 'right elbow', 'right wrist', 'right hand',
             'left hip', 'left knee', 'left ankle', 'left foot',
             'right hip', 'right knee', 'right ankle', 'right foot',
             'spine', 'tip of left hand', 'left thumb',
             'tip of right hand', 'right thumb']

shConnections = [
    (0, 2), (1, 3),      # Knöchel zu Knie
    (2, 4), (3, 5),      # Knie zu Hüfte
    (4, 5),              # Hüften verbinden
    (4, 10), (5, 11),    # Hüfte zu Schultern
    (10, 11),            # Schultern verbinden
    (10, 8), (11, 9),    # Schultern zu Ellbogen
    (8, 6), (9, 7),      # Ellbogen zu Handgelenken
    (10, 12), (11, 12)   # Kopf (vereinfacht)
]

ntuConnections = [
    (0, 1), (1, 20), (20, 2), (2, 3),  # Wirbelsäule bis Kopf
    (20, 4), (20, 8),          # Wirbelsäule zu Schultern
    (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),         # Linke Schulter zu Hand
    (8, 9), (9, 10), (10, 11), (11, 23), (11,24),       # Rechte Schulter zu Hand
    (0, 12), (0, 16),        # Wirbelsäule zu Hüften
    (12, 13), (13, 14), (14, 15),     # Linke Hüfte zu Fuß
    (16, 17), (17, 18), (18, 19)]     # Rechte Hüfte zu Fuß

njts = 13 # Number of joints in the Smarthome skeleton
ntunjts = 25 # Number of joints in the NTU skeleton
input_dir = "./Data/json"
output_dir = "./Data/preprocessed"

def process_all_jsons(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            #out_path = os.path.join(output_dir, filename.replace(".json", "_newFormat.json"))
            convert_skeleton(json_path)
            

            #np.save(out_path, seq)
            #print(f"✔️ {filename} → {os.path.basename(out_path)}")


def convert_skeleton(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    T = len(frames)
    coordniate_List = []

    for t, frame in enumerate(frames):
        if not frame:
            continue
        pose3d = frame[0]['pose3d']
        Xs = pose3d[:njts]
        Ys = pose3d[njts:2*njts]
        Zs = pose3d[2*njts:]

        formatedJoints = np.stack([Xs, Ys, Zs], axis=1).tolist()

        joints_coords = dict(zip(shJoints, formatedJoints))

        #print(joints_coords)
        plot_skeleton(formatedJoints, connections=shConnections, joint_names=shJoints)
        ntu_skeleton = SmartHomeToNTU(formatedJoints)
        coordniate_List.append(ntu_skeleton)
        #print(ntu_skeleton)
        plot_skeleton(ntu_skeleton, connections=ntuConnections, joint_names=ntuJoints)
    convert_coords_to_stgcn_npy(coordniate_List, f"{json_path.replace('.json', '_ntu.npy').replace('/json', '/preprocessed')}", V=ntunjts, M=1)


def SmartHomeToNTU(skeleton):
    """
    Convert a SmartHome skeleton to NTU format.
    """
    ntu_skeleton = np.zeros((ntunjts, 3))
    skeleton = np.array(skeleton)

    # Map SmartHome joints to NTU joints
    # First, map all existing joints
    ntu_skeleton[3] = skeleton[12]  # head
    ntu_skeleton[4] = skeleton[11]  # left shoulder
    ntu_skeleton[5] = skeleton[9]   # left elbow
    ntu_skeleton[6] = skeleton[7]   # left wrist
    ntu_skeleton[8] = skeleton[10]  # right shoulder
    ntu_skeleton[9] = skeleton[8]   # right elbow
    ntu_skeleton[10] = skeleton[6]  # right wrist
    ntu_skeleton[12] = skeleton[5]  # left hip
    ntu_skeleton[13] = skeleton[3]  # left knee
    ntu_skeleton[14] = skeleton[1]  # left ankle
    ntu_skeleton[16] = skeleton[4]  # right hip
    ntu_skeleton[17] = skeleton[2]  # right knee
    ntu_skeleton[18] = skeleton[0]  # right ankle
    # Then, fill in the with calculated joints
    # base of spine is the average of left and right hip
    ntu_skeleton[0] = (skeleton[4] + skeleton[5]) / 2 # base of spine 
    # spine is average of left and right shoulder
    ntu_skeleton[20] = (skeleton[10] + skeleton[11]) / 2  # spine
    # middle of spine is average of base of spine and spine
    ntu_skeleton[1] = (ntu_skeleton[0] + ntu_skeleton[20]) / 2  # middle of spine
    # neck 1/8 of the difrence between spine and head
    ntu_skeleton[2] = ntu_skeleton[20] + (ntu_skeleton[3] - ntu_skeleton[20]) / 7  # neck
    # left hand is the extension of the left wrist from the diection of the left elbow
    ntu_skeleton[7] = ntu_skeleton[6] + (ntu_skeleton[5] - ntu_skeleton[6]) / 4  # left hand
    # right hand is the extension of the right wrist from the diection of the right elbow
    ntu_skeleton[11] = ntu_skeleton[10] + (ntu_skeleton[9] - ntu_skeleton[10]) / 4  # right hand
    # left foot is the extension of the left ankle from the diection of the left knee
    ntu_skeleton[15] = ntu_skeleton[14] + (ntu_skeleton[14] - ntu_skeleton[13]) * 0.2  # left foot
    # right foot is the extension of the right ankle from the diection of the right knee
    ntu_skeleton[19] = ntu_skeleton[18] + (ntu_skeleton[18] - ntu_skeleton[17]) * 0.2  # right foot
    # left hand tip is the extension of the left hand from the diection of the left wrist
    ntu_skeleton[21] = ntu_skeleton[7] + (ntu_skeleton[6] - ntu_skeleton[7]) * 1.5  # tip of left hand
    # left thumb is the extension of the left hand from the diection of the left wrist
    ntu_skeleton[22] = ntu_skeleton[6] + (ntu_skeleton[7] - ntu_skeleton[6]) * 1  # left thumb
    # right hand tip is the extension of the right hand from the diection of the right wrist
    ntu_skeleton[23] = ntu_skeleton[11] + (ntu_skeleton[10] - ntu_skeleton[11]) * 1.5  # tip of right hand
    # right thumb is the extension of the right hand from the diection of the right wrist
    ntu_skeleton[24] = ntu_skeleton[10] + (ntu_skeleton[11] - ntu_skeleton[10]) * 1  # right thumb
    return ntu_skeleton.tolist()


def convert_coords_to_stgcn_npy(frames_list, output_path, V=25, M=1):
    """
    Konvertiert eine Liste von Frames in ST-GCN-kompatibles Format (C, T, V, M).

    Args:
        frames_list: Liste der Länge T, jedes Element ist eine Liste von V Gelenken [x,y,z].
                     Shape → (T, V, 3)
        output_path: Pfad zur Ausgabe .npy-Datei
        V: Anzahl der Gelenke pro Frame (default 25)
        M: Anzahl der Personen (default 1)
    """
    coords = np.array(frames_list, dtype=np.float32)
    if coords.ndim == 3:
        T, V_in, C = coords.shape
        if C != 3:
            raise ValueError(f"Letzte Dim muss 3 sein (x,y,z), ist aber {C}")
        if V_in != V:
            raise ValueError(f"Anzahl der Gelenke pro Frame muss {V}, ist aber {V_in}")

        # Transponieren zu (C, T, V)
        data = np.transpose(coords, (2, 0, 1))  # → (3, T, V)

        # Neue Achse für Personen: (C, T, V, M)
        data = data[..., np.newaxis]           # → (3, T, V, 1)

        # Speichern
        np.save(output_path, data)
        #print(f"Gespeichert: {output_path}, Shape: {data.shape}")
    else:
        print("Warnung: Eingabedaten haben unerwartete Dimensionen. Erwartet (T, V, C), aber erhalten:", coords.shape)
        print("For Path", output_path)
        print(coords)




def plot_skeleton(points, connections=None, joint_names=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Punkte plotten
    xs, ys, zs = zip(*points)
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # Gelenkverbindungen zeichnen (falls vorhanden)
    if connections:
        for i, j in connections:
            x = [points[i][0], points[j][0]]
            y = [points[i][1], points[j][1]]
            z = [points[i][2], points[j][2]]
            ax.plot(x, y, z, c='b')

    # Joint-Namen anzeigen (optional)
    if joint_names:
        for idx, (x, y, z) in enumerate(points):
            ax.text(x, y, z, joint_names[idx], fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Skeleton')
    plt.show()

 
#process_all_jsons(input_dir)
convert_skeleton("./Data/json/Cook.Cleandishes_p02_r00_v02_c03.json")
