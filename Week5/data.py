#data.py
import os
import glob
import torch
import numpy as np
import tqdm
from dataloader import read_file
from torch.utils.data import Dataset, DataLoader

#need to wrap in a PyTorch DataLoader
class JetClassDataset(Dataset):
    def __init__(self, particles, jets, labels):
        self.particles = torch.from_numpy(particles.transpose(0, 2, 1)).float()
        self.jets = torch.from_numpy(jets).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "particles": self.particles[idx],
            "jets": self.jets[idx],
            "labels": self.labels[idx]
        }
    
def get_dataloader(split= "train", # "val"/"test"
                   batch_size=32, 
                   shuffle=True, 
                   basedir="Week5/data", #folder with root files
                   max_files=None, 
                   target_class=None): # if int -> binary labels; if None -> multi-class
        
    split_map = {
        "train": "JetClass_Pythia_train_100M_part*",
        "val":   "JetClass_Pythia_val_5M",
        "test":  "JetClass_Pythia_test_20M"
    }
    if split not in split_map:
        raise ValueError(f"Invalid split '{split}'. Choose from {list(split_map.keys())}")
    
    # Construct the full search path and find the files
    search_path = os.path.join(basedir, split_map[split], "*.root")
    files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(f"No files found at '{search_path}'. Check your `basedir`.")

    if max_files is not None:
        files = files[:max_files]

    #loop over files and concatenate the data
    print(f"Loading {len(files)} files for '{split}' split...")
    all_particles = []
    all_jets = []
    all_labels = []
    for fp in files:
        print(f"  - Reading {os.path.basename(fp)}")
        xp, xj, y = read_file(fp)
        if target_class is not None:
            mask = np.argmax(y, axis=1) == target_class
            xp, xj, y = xp[mask], xj[mask], y[mask]
        else: 
            all_particles.append(xp)
            all_jets.append(xj)
            all_labels.append(y)
    
    xp_all = np.concatenate(all_particles, axis=0)
    xj_all = np.concatenate(all_jets, axis=0)
    y_all = np.concatenate(all_labels, axis=0)

    print(f"Loaded {len(y_all)} total jets.")

    if target_class is not None: #binary classification: 1 for target and 0 for others
        y_final = (np.argmax(y_all, axis=1) == target_class).astype(int)
    else: #multiclass: convert one-hot to class index bc crossentropyloss expects class indices
        y_final = np.argmax(y_all, axis=1)
    
    dataset = JetClassDataset(particles=xp_all, jets=xj_all, labels=y_final)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)