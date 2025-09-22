import os
import numpy as np


def giveShapeOfNpyFiles(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            npy_path = os.path.join(input_dir, filename)
            data = np.load(npy_path)
            print(f"Shape of {filename}: {data.shape}")



# giveShapeOfNpyFiles("resources/text_feats/pku51_split3/clip-vit-b-32")
# giveShapeOfNpyFiles("resources/sk_feats/stgcn_pku51_split1_5_r")
# print("-------------------------------------")
giveShapeOfNpyFiles("resources/sk_feats/stgcn_tsh31_split1_val_5_r")
# giveShapeOfNpyFiles("resources/label_splits/tsh31_split1")