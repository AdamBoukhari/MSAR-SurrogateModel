import numpy as np
import time
import matplotlib.pyplot as plt

from ModelLoader import SimulatorLoader, SurrogateLoader
import torch

import os
import json
from datetime import datetime

MAX_SRU = 5

def compute_bounds(sim_phys):
    traj = sim_phys.trajectory
    xs = traj[:, :, 0]
    ys = traj[:, :, 1]

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    size = max(max_x - min_x, max_y - min_y) / 2 + 0.5

    return (center_x - size, center_x + size, center_y - size, center_y + size)


def generate_batch(bounds, batch_size, active_sru, timestamps, rng):
    min_x, max_x, min_y, max_y = bounds
    batch = []

    for _ in range(batch_size):
        srus = []
        for i in range(active_sru):
            px = rng.uniform(min_x, max_x)
            py = rng.uniform(min_y, max_y)
            srus.append({
                "position": (px, py),
                "time": timestamps[i]
            })
        batch.append(srus)

    return batch
def generate_sru_tensor_fast(bounds, batch_size, active_sru, timestamps, rng, start_lon, start_lat, device):
    min_x, max_x, min_y, max_y = bounds

    xs = rng.uniform(min_x, max_x, size=(batch_size, active_sru))
    ys = rng.uniform(min_y, max_y, size=(batch_size, active_sru))

    ts = np.array(timestamps)[None, :]  # (1, n_sru)
    ts = np.repeat(ts, batch_size, axis=0)  # (batch, n_sru)

    tensor = np.zeros((1, batch_size, active_sru, 5), dtype=np.float32)

    tensor[0, :, :, 0] = xs - start_lon
    tensor[0, :, :, 1] = ys - start_lat
    tensor[0, :, :, 2] = (ts - (2*3600)) / (40*3600)
    tensor[0, :, :, 3] = ys / 90.0
    tensor[0, :, :, 4] = 1.0

    return torch.from_numpy(tensor).to(device)


def naive_random_search(sim, bounds, active_sru, timestamps, time_budget, rng):
    history = []  # (time, best)
    best = -np.inf

    start = time.time()

    while True:
        now = time.time()
        elapsed = now - start
        if elapsed >= time_budget:
            break

        batch = generate_batch(bounds, 50, active_sru, timestamps, rng)
        scores = sim.get_pos_batch(batch)

        batch_best = max(scores)
        best = max(best, batch_best)

        history.append((elapsed, best))

    return history


# def hybrid_search(sim, surrogate, bounds, active_sru, timestamps, time_budget, rng):
#     history = []  # (time, best)
#     best = -np.inf

#     start = time.time()

#     while True:
#         now = time.time()
#         elapsed = now - start
#         if elapsed >= time_budget:
#             break

#         # 1. large random batch
#         batch = generate_batch(bounds, 1000, active_sru, timestamps, rng)

#         # 2. surrogate eval
#         scores = surrogate.get_pos_batch(batch)
#         scores = np.array(scores)

#         # 3. select top 50
#         top_idx = np.argpartition(scores, -50)[-50:]
#         top_batch = [batch[i] for i in top_idx]

#         # 4. simulator eval
#         sim_scores = sim.get_pos_batch(top_batch)

#         batch_best = max(sim_scores)
#         best = max(best, batch_best)

#         history.append((elapsed, best))

#     return history

def load_drift_files(folder_path):
    files = []
    for f in os.listdir(folder_path):
        if f.endswith(".npz"):
            files.append(os.path.join(folder_path, f))
    files.sort()
    return files

def hybrid_search(sim, surrogate, bounds, active_sru, timestamps, time_budget, rng):
    history = []
    best = -np.inf

    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed >= time_budget:
            break

        tensor = generate_sru_tensor_fast(
            bounds,
            1000,
            active_sru,
            timestamps,
            rng,
            surrogate.start_lon,
            surrogate.start_lat,
            surrogate.device
        )

        with torch.no_grad():
            pred = surrogate.model(
                surrogate.drift_tensor,
                tensor,
                drift_tokens=surrogate.token
            )

        scores = pred.squeeze(0)  # torch tensor

        top_scores, top_idx = torch.topk(scores, 50)

        top_idx = top_idx.cpu().numpy()

        top_batch = []

        for idx in top_idx:
            srus = []
            for i in range(active_sru):

                x = tensor[0, idx, i, 0].item() + surrogate.start_lon
                y = tensor[0, idx, i, 1].item() + surrogate.start_lat
                t = timestamps[i]

                srus.append({
                    "position": (x, y),
                    "time": t
                })

            top_batch.append(srus)

        sim_scores = sim.get_pos_batch(top_batch)

        batch_best = max(sim_scores)
        best = max(best, batch_best)

        history.append((elapsed, best))

    return history

def run_experiment_multi(drift_folder, runs=5, time_budget=1.0, active_sru=3):

    drift_files = load_drift_files(drift_folder)

    all_naive = []
    all_hybrid = []

    for d_idx, drift_path in enumerate(drift_files):
        print(f"\n=== Drift {d_idx+1}/{len(drift_files)} ===")

        sim = SimulatorLoader("")
        surrogate = SurrogateLoader("output/best_model.pt")

        sim.load_drift(drift_path)
        surrogate.load_drift(drift_path)

        bounds = compute_bounds(sim)

        for r in range(runs):
            print(f"  Run {r+1}/{runs}")

            rng = np.random.RandomState(r)  # important: reset per run

            timestamps = [rng.uniform(6*3600, 40*3600) for _ in range(active_sru)]

            init_srus = [{"position": {"x":0, "y":0}, "time":0} for _ in range(active_sru)]
            sim.create_srus(init_srus)
            surrogate.create_srus(init_srus)

            naive_scores = naive_random_search(
                sim, bounds, active_sru, timestamps, time_budget, rng
            )

            hybrid_scores = hybrid_search(
                sim, surrogate, bounds, active_sru, timestamps, time_budget, rng
            )

            all_naive.append(naive_scores)
            all_hybrid.append(hybrid_scores)

    return all_naive, all_hybrid


def plot_results(naive_all, hybrid_all, folder, resolution=100):

    max_time = min(
        max(run[-1][0] for run in naive_all),
        max(run[-1][0] for run in hybrid_all)
    )

    grid = np.linspace(0, max_time, resolution)

    def interpolate_runs(runs):
        interp = []
        for run in runs:
            times = np.array([t for t, _ in run])
            values = np.array([v for _, v in run])

            res = np.zeros_like(grid)
            j = 0
            current = values[0]

            for i, g in enumerate(grid):
                while j < len(times) and times[j] <= g:
                    current = values[j]
                    j += 1
                res[i] = current

            interp.append(res)

        return np.array(interp)

    naive_interp = interpolate_runs(naive_all)
    hybrid_interp = interpolate_runs(hybrid_all)

    naive_mean = naive_interp.mean(axis=0)
    hybrid_mean = hybrid_interp.mean(axis=0)

    plt.figure()
    plt.plot(grid, naive_mean, label="Naive Simulator (50 batch)")
    plt.plot(grid, hybrid_mean, label="Hybrid (1000 surrogate -> top50 sim)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Best Score")
    plt.legend()
    plt.title("Convergence over Real Time")

    plt.savefig(os.path.join(folder, "convergence.png"), dpi=300)

    plt.show()

def create_experiment_folder(runs, time_budget, active_sru):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    folder_name = f"exp_runs{runs}_time{time_budget}_sru{active_sru}_{now}"
    os.makedirs(folder_name, exist_ok=True)

    return folder_name

def save_config(folder, drift_path, runs, time_budget, active_sru):
    config = {
        "drift_path": drift_path,
        "runs": runs,
        "time_budget": time_budget,
        "active_sru": active_sru
    }

    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

def save_raw_results(folder, naive_all, hybrid_all):
    data = {
        "naive": naive_all,
        "hybrid": hybrid_all
    }

    with open(os.path.join(folder, "results.json"), "w") as f:
        json.dump(data, f)
def create_experiment_folder_multi(runs, time_budget, active_sru, n_files):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    folder_name = f"exp_MULTI_files{n_files}_runs{runs}_time{time_budget}_sru{active_sru}_{now}"
    os.makedirs(folder_name, exist_ok=True)

    return folder_name

if __name__ == "__main__":

    drift_folder = r"dataset\DriftReduced\validation\tung"
    runs = 20
    time_budget = 2.0
    active_sru = 5

    drift_files = load_drift_files(drift_folder)

    folder = create_experiment_folder_multi(
        runs, time_budget, active_sru, len(drift_files)
    )

    save_config(folder, drift_folder, runs, time_budget, active_sru)

    naive_all, hybrid_all = run_experiment_multi(
        drift_folder=drift_folder,
        runs=runs,
        time_budget=time_budget,
        active_sru=active_sru
    )

    save_raw_results(folder, naive_all, hybrid_all)

    plot_results(naive_all, hybrid_all, folder)