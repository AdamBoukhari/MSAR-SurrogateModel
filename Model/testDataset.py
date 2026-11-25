import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import DriftDataset, DriftValidationDataset
import os 
import matplotlib.pyplot as plt
from Model.model import *

BATCH_STRATEGIES = 32
EPOCHS = 50
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DRIFT_TRAIN = "dataset/DriftReduced/train"
DRIFT_VAL = "dataset/DriftReduced/validation"

IO_TRAIN = "dataset/InputOutput/train"
IO_VAL = "dataset/InputOutput/validation"

SAVE_DIR = "output"

def plot_dataset_distribution(loader, num_batches=100):
    all_labels = []
    
    print(f"Collecte des labels sur {num_batches} batches...")
    for i, (_, _, batch_outputs) in enumerate(tqdm(loader)):
        if i >= num_batches:
            break
        all_labels.extend(batch_outputs.numpy().flatten())
    
    all_labels = np.array(all_labels)
    
    mean = np.mean(all_labels)
    std = np.std(all_labels)
    median = np.median(all_labels)
    zeros_pc = (all_labels < 0.01).sum() / len(all_labels) * 100

    plt.figure(figsize=(10, 6))
    plt.hist(all_labels, bins=100, color='skyblue', edgecolor='black', alpha=0.8)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Moyenne: {mean:.3f}')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Médiane: {median:.3f}')
    
    plt.title(f"Distribution des Probabilités (Labels)\nStd: {std:.3f} | 'Zéro' (<1%): {zeros_pc:.1f}%")
    plt.xlabel("Valeur du label (0.0 à 1.0)")
    plt.ylabel("Nombre d'occurrences")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

from torch.utils.data import DataLoader
train_dataset = DriftDataset(DRIFT_TRAIN, IO_TRAIN, batch_strategies=BATCH_STRATEGIES)
loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
plot_dataset_distribution(loader, num_batches=200)