import csv
import numpy as np
import os

INPUTDIR = "Data/json"  # <-- Pfad zu deinem Ordner mit JSON-Dateien
INPUTFILE = "class_lists/toyota51.csv"  # <-- Pfad zur CSV-Datei mit den Labels
OUTPUTDIR = "resources/label_splits"  # <-- Zielordner für die .npy-Dateien

split1 = [0, 1,2,3,4]
split1_val = [5,6,7,8,9]

split2 = [1, 17, 20, 33, 45]
split2_val = [21, 25, 28, 34, 29]

split3 = [15, 16, 17, 18, 50]
split3_val = [22, 23, 24, 25, 26]

splits = {
    "toyota51_split1": {
        "test": split1,
        "val": split1_val
    },
    "toyota51_split2": {
        "test": split2,
        "val": split2_val
    },
    "toyota51_split3": {
        "test": split3,
        "val": split3_val
    }
}

def create_label_splits(input_file, output_dir):
    csv_file = input_file
    for split_name, split_data in splits.items():
        test_indices = split_data["test"]
        val_indices = split_data["val"]
        
        rs41_0 = []
        rs46 = []
        ru5 = []
        rv5_0 = []

        with open(csv_file, 'r') as f:

        # Create directories for each split
            outDir = os.path.join(output_dir, split_name)
            os.makedirs(outDir, exist_ok=True)
            for idx, label in enumerate(f):
                if idx == 0:
                    continue
                idx = int(idx) - 1  # Adjust index to match the split indices
            # Process test set
                label = label.strip()
                label = label.split(",")[1]
                id = test_indices.index(idx) if idx in test_indices else None
                id_val = val_indices.index(idx) if idx in val_indices else None
                
                if  id != None and id_val != None:
                    print(f"Error: Test and validation indices overlap for index {idx}.")
                elif id != None:
                    ru5.append(label)
                elif id_val != None:
                    rv5_0.append(label)
                    rs46.append(label)
                else:
                    rs46.append(label)
                    rs41_0.append(label)

        # Save the lists as .npy files in the corresponding outDir
        np.save(os.path.join(outDir, "rs41_0.npy"), np.array(rs41_0))
        np.save(os.path.join(outDir, "rs46.npy"), np.array(rs46))
        np.save(os.path.join(outDir, "ru5.npy"), np.array(ru5))
        np.save(os.path.join(outDir, "rv5_0.npy"), np.array(rv5_0))
    

def giveShapeOfNpyFiles(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            npy_path = os.path.join(input_dir, filename)
            data = np.load(npy_path)
            print(f"Shape of {filename}: {data.shape}")

giveShapeOfNpyFiles("resources/sk_feats/stgcn_ntu60_split2_5_r")
#create_label_splits(INPUTFILE, OUTPUTDIR)