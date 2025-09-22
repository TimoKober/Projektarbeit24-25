import os
import json
import numpy as np

visualEncoder = "stgcn"
INPUTDIR = "Data/json"  # <-- Pfad zu deinem Ordner mit JSON-Dateien
INPUTFILE = "class_lists/toyota51.csv"  # <-- Pfad zur CSV-Datei mit den Labels
OUTPUTDIR = "resources/label_splits"  # <-- Zielordner für die .npy-Dateien
input_dir = "./Data/json"

LABELS = ['Cook.Cleandishes','Cook.Cleanup','Cook.Cut', 'Cook.Stir', 'Cook.Usestove', 
'Cutbread', 'Drink.Frombottle', 'Drink.Fromcan', 'Drink.Fromcup', 'Drink.Fromglass',
'Eat.Attable', 'Eat.Snack', 'Enter','Getup', 'Laydown', 'Leave', 
'Makecoffee.Pourgrains', 'Makecoffee.Pourwater', 'Maketea.Boilwater', 
'Maketea.Insertteabag', 'Pour.Frombottle', 
'Pour.Fromcan', 'Pour.Fromkettle', 'Readbook', 'Sitdown', 'Takepills', 
'Uselaptop','Usetablet', 'Usetelephone', 'Walk', 'WatchTV']

split_definitions = {
    "split1": ['Drink.Frombottle', 'Makecoffee.Pourwater','Pour.Fromcan', 'Readbook', 'Uselaptop'],
    "split2": ['Cook.Cleandishes','Cook.Cleanup','Cook.Cut', 'Cook.Stir', 'Cook.Usestove'],
    "split3": ['Eat.Attable','Eat.Snack','Sitdown','Takepills','WatchTV']
}

# Lade Features und Labels
X = np.load("Data/features/smarthome_train.npy")
y = np.load("Data/features/smarthome_train_label.npy")
Y_onehot = np.load("Data/features/smarthome_train_out.npy")

label2idx = {label: idx for idx, label in enumerate(LABELS)}

# Hilfsfunktion um Splits zu erstellen
def create_split(split_name, test_actions, base_dir="resources/sk_feats", encoder="stgcn", dataset="tsh31", size=5):
    test_indices = [label2idx[a] for a in test_actions]
    mask_test = np.isin(y, test_indices)
    mask_train = ~mask_test

    # Splits
    X_train, y_train, Y_train = X[mask_train], y[mask_train], Y_onehot[mask_train]
    X_test, y_test, Y_test = X[mask_test], y[mask_test], Y_onehot[mask_test]

    seen_indices = [idx for idx, label in enumerate(LABELS) if label not in test_actions]
    val_actions = np.random.choice(seen_indices, size=5, replace=False)

    mask_val = np.isin(y_train, val_actions)
    mask_train_final = ~mask_val
    
    X_val, y_val, Y_val = X_train[mask_val], y_train[mask_val], Y_train[mask_val]
    X_train, y_train, Y_train = X_train[mask_train_final], y_train[mask_train_final], Y_train[mask_train_final]

    # Generalized Testdaten = alles (Train+Val+Test)
    seen_for_gtest = np.random.choice([idx for idx in seen_indices if idx not in val_actions], size=5, replace=False)
    mask_gtest = np.isin(y, np.concatenate([test_indices, seen_for_gtest]))

    X_gtest, y_gtest, Y_gtest = X[mask_gtest], y[mask_gtest], Y_onehot[mask_gtest]

    # Ordnerpfad
    split_dir = f"{base_dir}/{encoder}_{dataset}_{split_name}_{size}_r"
    os.makedirs(split_dir, exist_ok=True)

    # Validation-Ordnerpfad
    val_dir = f"{base_dir}/{encoder}_{dataset}_{split_name}_val_{size}_r"
    os.makedirs(val_dir, exist_ok=True)

    # Speichern im Split-Ordner
    np.save(os.path.join(split_dir, "train.npy"), X_train)
    np.save(os.path.join(split_dir, "train_label.npy"), y_train)
    np.save(os.path.join(split_dir, "train_out.npy"), Y_train)

    np.save(os.path.join(split_dir, "val.npy"), X_val)
    np.save(os.path.join(split_dir, "val_label.npy"), y_val)
    np.save(os.path.join(split_dir, "val_out.npy"), Y_val)

    np.save(os.path.join(split_dir, "ztest.npy"), X_test)
    np.save(os.path.join(split_dir, "z_label.npy"), y_test)
    np.save(os.path.join(split_dir, "ztest_out.npy"), Y_test)

    np.save(os.path.join(split_dir, "gtest.npy"), X_gtest)
    np.save(os.path.join(split_dir, "g_label.npy"), y_gtest)
    np.save(os.path.join(split_dir, "gtest_out.npy"), Y_gtest)

    # Gleiche Dateien zusätzlich im Validation-Ordner speichern
    np.save(os.path.join(val_dir, "train.npy"), X_train)
    np.save(os.path.join(val_dir, "train_label.npy"), y_train)
    np.save(os.path.join(val_dir, "train_out.npy"), Y_train)

    np.save(os.path.join(val_dir, "val.npy"), X_val)
    np.save(os.path.join(val_dir, "val_label.npy"), y_val)
    np.save(os.path.join(val_dir, "val_out.npy"), Y_val)

    np.save(os.path.join(val_dir, "ztest.npy"), X_test)
    np.save(os.path.join(val_dir, "z_label.npy"), y_test)
    np.save(os.path.join(val_dir, "ztest_out.npy"), Y_test)

    np.save(os.path.join(val_dir, "gtest.npy"), X_gtest)
    np.save(os.path.join(val_dir, "g_label.npy"), y_gtest)
    np.save(os.path.join(val_dir, "gtest_out.npy"), Y_gtest)

    label_dir = f"resources/label_splits/tsh31_{split_name}"
    os.makedirs(label_dir, exist_ok=True)

    np.save(os.path.join(label_dir, "rs26.npy"), np.array(seen_indices, dtype=np.int32))
    np.save(os.path.join(label_dir, "ru5.npy"), np.array(test_indices, dtype=np.int32))
    np.save(os.path.join(label_dir, "rv5_0.npy"), np.array(val_actions, dtype=np.int32))

    rs_without_val_test = [idx for idx in seen_indices if idx not in val_actions]
    np.save(os.path.join(label_dir, "rs21_0.npy"), np.array(rs_without_val_test, dtype=np.int32))

    print(f"{split_name} done. Train {X_train.shape}, Test {X_test.shape}, Val {X_val.shape}, GTest {X_gtest.shape}")

# Alle Splits erstellen
for split_name, actions in split_definitions.items():
    create_split(split_name, actions)