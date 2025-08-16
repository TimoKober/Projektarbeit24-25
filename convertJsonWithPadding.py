import os
import json
import numpy as np

INPUT_DIR = "Data/json"    # <-- Pfad zu deinem Ordner mit JSON-Dateien
OUTPUT_DIR = "resources/preprocessed/npy"  # <-- Zielordner für die .npy-Dateien
TARGET_LEN = 300               # Zielanzahl an Frames
V = 25                         # Anzahl Gelenke
C = 3                          # x, y, z
M = 1                          # Personenanzahl (hier immer 1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_pose3d_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    T = len(frames)

    skeleton = np.zeros((C, T, V, M), dtype=np.float32)

    for t, frame in enumerate(frames):
        if not frame:
            continue
        pose3d = frame[0]['pose3d']
        joints = np.array(pose3d).reshape(V, C)
        for c in range(C):
            skeleton[c, t, :, 0] = joints[:, c]
    
    return skeleton

def compute_motion_energy(seq):
    # Differenz zwischen aufeinanderfolgenden Frames
    vel = np.linalg.norm(seq[:, 1:, :, :] - seq[:, :-1, :, :], axis=0)  # (T-1, V, M)
    energy = np.sum(vel, axis=(1, 2))  # (T-1,)
    return energy

def truncate_or_pad(seq, target_len=150):
    T = seq.shape[1]

    if T == target_len:
        return seq

    elif T < target_len:
        # Zero Padding
        pad = np.zeros((C, target_len - T, V, M), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=1)

    else:
        # Sliding window mit höchster Bewegungsenergie finden
        energy = compute_motion_energy(seq)
        window_scores = np.array([
            np.sum(energy[i:i + target_len - 1])
            for i in range(T - target_len + 1)
        ])
        start = np.argmax(window_scores)
        return seq[:, start:start + target_len, :, :]

def process_all_jsons(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename.replace(".json", ".npy"))

            seq = extract_pose3d_from_json(json_path)
            seq_processed = truncate_or_pad(seq, TARGET_LEN)

            np.save(out_path, seq_processed)
            print(f"✔️ {filename} → {os.path.basename(out_path)}")

process_all_jsons(INPUT_DIR, OUTPUT_DIR)
