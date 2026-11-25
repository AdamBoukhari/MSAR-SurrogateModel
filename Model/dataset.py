import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import math
import numpy as np
from datetime import datetime

class DriftDataset(Dataset):

    def __init__(self, drift_path, io_path, batch_strategies=32, sampling_repeat=4, drift_size=256):
        self.batch_strategies = batch_strategies
        self.sampling_repeat = sampling_repeat
        self.drift_path = drift_path
        self.io_path = io_path

        self.drift_size = drift_size

        self.files = sorted(os.listdir(drift_path))


    def __len__(self):
        return len(self.files) * self.sampling_repeat

    def load_drift_file(self, drift_file):
        data = np.load(drift_file, allow_pickle=True, mmap_mode="r")
        trajectory = data["trajectory"]      # (N, T, 2)
        start_lon = float(data["start_lon"])
        start_lat = float(data["start_lat"])
        start_timestamp_str = str(data["timestamp"])
        start_timestamp_int = int(datetime.fromisoformat(start_timestamp_str).timestamp())
        return trajectory, start_lon, start_lat, start_timestamp_int

    def sample_trajectories(self, trajectory, start_lon, start_lat):
        N = trajectory.shape[0]
        if N >= self.drift_size:
            idx = np.random.choice(N, self.drift_size, replace=False)
        else:
            idx = np.random.choice(N, self.drift_size, replace=True)

        traj_sampled = trajectory[idx].copy()
        traj_sampled[..., 0] -= start_lon
        traj_sampled[..., 1] -= start_lat

        T = traj_sampled.shape[1]
        if T < 577:
            last = traj_sampled[:, -1:, :]                      # (N, 1, 2)
            pad = np.repeat(last, 577 - T, axis=1)              # (N, 577-T, 2)
            traj_sampled = np.concatenate([traj_sampled, pad], axis=1)


        return traj_sampled

    def sample_sru_batch(self, io_file):
        io_data = np.load(io_file, mmap_mode="r")
        positions = io_data["positions"]
        times     = io_data["times"]
        mask      = io_data["mask"]
        outputs   = io_data["outputs"]
        
        n = positions.shape[0]
        stride = math.ceil(n / self.batch_strategies)
        offset = np.random.randint(0, stride)
        indices = (offset + np.arange(self.batch_strategies) * stride) % n

        return positions[indices], times[indices], mask[indices], outputs[indices]
    
    
    def normalize_sru_batch(self, positions, times, mask, start_lon, start_lat, max_lat_theorique=90.0):
        batch_size, n_sru, _ = positions.shape

        pos_rel = positions.copy()
        pos_rel[..., 0] -= start_lon
        pos_rel[..., 1] -= start_lat

        times_norm = ( times-(2*3600)) / (40 * 3600)  

        lat_norm = (positions[..., 1]) / max_lat_theorique

        normalized_batch = np.zeros((batch_size, n_sru, 5), dtype=np.float32)
        normalized_batch[..., 0] = pos_rel[..., 0]        # x
        normalized_batch[..., 1] = pos_rel[..., 1]        # y
        normalized_batch[..., 2] = times_norm            # temps
        normalized_batch[..., 3] = lat_norm              # hauteur verticale
        normalized_batch[..., 4] = mask                  # mask

        return torch.from_numpy(normalized_batch)

    def __getitem__(self, idx):
        file = self.files[idx % len(self.files)]
        drift_file = os.path.join(self.drift_path, file)
        io_file = os.path.join(self.io_path, file.replace(".npz", ".dataset.npz"))

        trajectory, start_lon, start_lat, time_ts = self.load_drift_file(drift_file)
        traj_sampled = self.sample_trajectories(trajectory, start_lon, start_lat)
        batch_positions, batch_times, batch_mask, batch_outputs = self.sample_sru_batch(io_file)

        batch_inputs = self.normalize_sru_batch(batch_positions, batch_times, batch_mask, start_lon, start_lat)

        return (
            torch.from_numpy(traj_sampled).float(),
            batch_inputs.clone().detach().float(), 
            torch.from_numpy(batch_outputs).float()
        )

class DriftValidationDataset(Dataset):
    def __init__(self, drift_path, io_path, batch_strategies=32, drift_size=256):
        self.drift_path = drift_path
        self.io_path = io_path
        self.batch_strategies = batch_strategies

        self.files = sorted(os.listdir(drift_path))
        self.drift_size = drift_size 
        
        # Préparer tous les indices
        self.indices = []
        for f in self.files:      
            io_data = np.load(os.path.join(io_path, f.replace(".npz", ".dataset.npz")), mmap_mode="r")
            n = io_data["positions"].shape[0]
            stride = math.ceil(n / batch_strategies)
            for offset in range(stride):
                self.indices.append((f, offset))

    def __len__(self):
        return len(self.indices)

    def load_drift_file(self, drift_file):
        data = np.load(drift_file, allow_pickle=True, mmap_mode="r")
        trajectory = data["trajectory"]      # (N, T, 2)
        start_lon = float(data["start_lon"])
        start_lat = float(data["start_lat"])
        start_timestamp_str = str(data["timestamp"])
        start_timestamp_int = int(datetime.fromisoformat(start_timestamp_str).timestamp())
        return trajectory, start_lon, start_lat, start_timestamp_int

    def sample_trajectories(self, trajectory, start_lon, start_lat):
        N = trajectory.shape[0]
        if N >= self.drift_size:
            idx = np.random.choice(N, self.drift_size, replace=False)
        else:
            idx = np.random.choice(N, self.drift_size, replace=True)

        traj_sampled = trajectory[idx].copy()
        traj_sampled[..., 0] -= start_lon
        traj_sampled[..., 1] -= start_lat
        
        T = traj_sampled.shape[1]
        if T < 577:
            last = traj_sampled[:, -1:, :]                      # (N, 1, 2)
            pad = np.repeat(last, 577 - T, axis=1)              # (N, 577-T, 2)
            traj_sampled = np.concatenate([traj_sampled, pad], axis=1)


        return traj_sampled
    
    def normalize_sru_batch(self, positions, times, mask, start_lon, start_lat, max_lat_theorique=90.0):
        batch_size, n_sru, _ = positions.shape

        pos_rel = positions.copy()
        pos_rel[..., 0] -= start_lon
        pos_rel[..., 1] -= start_lat

        times_norm = ( times-(2*3600)) / (40 * 3600)  

        lat_norm = (positions[..., 1]) / max_lat_theorique

        normalized_batch = np.zeros((batch_size, n_sru, 5), dtype=np.float32)
        normalized_batch[..., 0] = pos_rel[..., 0]        # x
        normalized_batch[..., 1] = pos_rel[..., 1]        # y
        normalized_batch[..., 2] = times_norm            # temps
        normalized_batch[..., 3] = lat_norm              # hauteur verticale
        normalized_batch[..., 4] = mask                  # mask

        return torch.from_numpy(normalized_batch)    

    def __getitem__(self, idx):
        file, offset = self.indices[idx]
        drift_file = os.path.join(self.drift_path, file)
        io_file = os.path.join(self.io_path, file.replace(".npz", ".dataset.npz"))

        # load drift
        trajectory, start_lon, start_lat, start_ts = self.load_drift_file(drift_file)  
        traj_sampled = self.sample_trajectories(trajectory, start_lon, start_lat)      

        # load batch SRU déterministe
        io_data = np.load(io_file, mmap_mode="r")
        n = io_data["positions"].shape[0]
        stride = math.ceil(n / self.batch_strategies)
        indices = (offset + np.arange(self.batch_strategies) * stride) % n

        positions = io_data["positions"][indices]
        times     = io_data["times"][indices]
        mask      = io_data["mask"][indices]
        outputs   = io_data["outputs"][indices]

        batch_inputs = self.normalize_sru_batch(positions, times, mask, start_lon, start_lat)

        return (
            torch.from_numpy(traj_sampled).float(), 
            batch_inputs.clone().detach().float(), 
            torch.from_numpy(outputs).float()
        )