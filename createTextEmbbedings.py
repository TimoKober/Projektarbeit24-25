from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
import os

modelName = 'clip-vit-b-32'  # Model name for CLIP
moPath = 'resources/text_feats/tsh31_split1'  # Path to the .npy file

OUTPUTDIR = "resources/text_feats/tsh31_split1"  # <-- Zielordner für die .npy-Dateien


actions = ['Cook.Cleandishes','Cook.Cleanup','Cook.Cut', 'Cook.Stir', 'Cook.Usestove', 
'Cutbread', 'Drink.Frombottle', 'Drink.Fromcan', 'Drink.Fromcup', 'Drink.Fromglass',
'Eat.Attable', 'Eat.Snack', 'Enter','Getup', 'Laydown', 'Leave', 
'Makecoffee.Pourgrains', 'Makecoffee.Pourwater', 'Maketea.Boilwater', 
'Maketea.Insertteabag', 'Pour.Frombottle', 
'Pour.Fromcan', 'Pour.Fromkettle', 'Readbook', 'Sitdown', 'Takepills', 
'Uselaptop','Usetablet', 'Usetelephone', 'Walk', 'WatchTV']

# 31 actions from toyota51.csv, without the sub-actions

def create_text_embeddings(actions, modelName, outputPath):

    outputPath = f"{outputPath}/{modelName}"
    #Load CLIP model
    model = SentenceTransformer(modelName)
    #Encode text descriptions
    text_emb = model.encode(actions, convert_to_tensor=True)
    text_emb = text_emb.cpu().numpy()
    os.makedirs(outputPath, exist_ok=True)

    np.save(os.path.join(outputPath, f"lb_{len(actions)}.npy"), np.array(text_emb))

    print(text_emb.shape)  

create_text_embeddings(actions, modelName, OUTPUTDIR)
# create_text_embeddings(actions_short, modelName, modelPath)





