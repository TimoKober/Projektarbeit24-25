from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
import os

modelName = 'clip-vit-b-32'  # Model name for CLIP
moPath = 'resources/text_feats/toyota51'  # Path to the .npy file

OUTPUTDIR = "resources/text_feats/toyota51"  # <-- Zielordner für die .npy-Dateien


actions = [
    "enter",
    "walk",
    "make_coffee",
    "get_water",
    "make_coffee.pour_water",
    "use_drawer",
    "make_coffee.pour_grains",
    "use_telephone",
    "leave",
    "put_something_on_table",
    "take_something_off_table",
    "pour.from_kettle",
    "stir_coffee/tea",
    "drink.from_cup",
    "dump_in_trash",
    "make_tea",
    "make_tea.boil_water",
    "use_cupboard",
    "insert_tea_bag",
    "read",
    "take_pills",
    "use_fridge",
    "clean_dishes",
    "clean_dishes.put_something_in_sink",
    "eat_snack",
    "sit_down",
    "watch_tv",
    "use_laptop",
    "get_up",
    "drink.from_bottle",
    "pour.from_bottle",
    "drink.from_glass",
    "lay_down",
    "drink.from_can",
    "write",
    "breakfast",
    "breakfast.spread_jam_or_butter",
    "breakfast.cut_bread",
    "breakfast.eat_at_table",
    "breakfast.take_ham",
    "clean_dishes.dry_up",
    "wipe_table",
    "cook",
    "cook.cut",
    "cook.use_stove",
    "cook.stir",
    "cook.use_oven",
    "clean_dishes.clean_with_water",
    "use_tablet",
    "use_glasses",
    "pour.from_can"
]

# 31 actions from toyota51.csv, without the sub-actions
actions_short = [
    "enter",
    "walk",
    "make coffee",
    "get water",
    "use drawer",
    "use telephone",
    "leave",
    "put something on table",
    "take something off table",
    "pour",
    "stir coffee/tea",
    "dump in trash",
    "make tea",
    "use cupboard",
    "insert tea bag",
    "read",
    "take pills",
    "use fridge",
    "clean dishes",
    "eat snack",
    "sit down",
    "watch tv",
    "use laptop",
    "get up",
    "drink",
    "lay down",
    "write",
    "breakfast",
    "wipe table",
    "cook",
    "use tablet",
    "use glasses",
]

def create_text_embeddings(actions, modelName, outputPath):

    outputPath = f"{outputPath}{len(actions)}/{modelName}"
    #Load CLIP model
    model = SentenceTransformer(modelName)
    #Encode text descriptions
    text_emb = model.encode(actions, convert_to_tensor=True)
    text_emb = text_emb.cpu().numpy()
    os.makedirs(outputPath, exist_ok=True)

    np.save(os.path.join(outputPath, f"lb_{len(actions)}.npy"), np.array(text_emb))

    print(text_emb.shape)  # Output: (3, 512)

create_text_embeddings(actions_short, modelName, f"resources/text_feats/toyota")
# create_text_embeddings(actions_short, modelName, modelPath)





