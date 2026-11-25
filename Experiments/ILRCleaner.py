import numpy as np
import time
import matplotlib.pyplot as plt
from ModelLoader import SimulatorLoader, SurrogateLoader
import torch
import os
import json
from datetime import datetime
import cma
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

def generate_candidates(solution, bounds, batch_size, active_sru, rng, std_scale=0.2):
    min_x, max_x, min_y, max_y = bounds
    std_x = std_scale * (max_x - min_x)
    std_y = std_scale * (max_y - min_y)

    sol_x = np.array([s["position"][0] for s in solution])
    sol_y = np.array([s["position"][1] for s in solution])

    noise = rng.normal(0, 1, size=(batch_size, active_sru, 2))
    
    xs = sol_x[None, :] + noise[:, :, 0] * std_x
    ys = sol_y[None, :] + noise[:, :, 1] * std_y

    xs = np.clip(xs, min_x, max_x)
    ys = np.clip(ys, min_y, max_y)

    return xs, ys

def build_tensor_from_candidates(xs, ys, timestamps, start_lon, start_lat, device):
    batch_size = xs.shape[0]
    active_sru = xs.shape[1]
    
    ts = np.array(timestamps)[None, :]
    ts = np.repeat(ts, batch_size, axis=0)
    
    tensor = np.zeros((1, batch_size, MAX_SRU, 5), dtype=np.float32)
    tensor[0, :, :active_sru, 0] = xs - start_lon
    tensor[0, :, :active_sru, 1] = ys - start_lat
    tensor[0, :, :active_sru, 2] = (ts - (2*3600)) / (40*3600)
    tensor[0, :, :active_sru, 3] = ys / 90.0
    tensor[0, :, :active_sru, 4] = 1.0 
    tensor[0, :, active_sru:, 4] = 0.0

    return torch.from_numpy(tensor).to(device)


def evaluate_classic(xs, ys, timestamps, active_sru, iteration_budget, sim):
    """Évalue tout le batch avec le simulateur."""
    batch = []
    for b in range(iteration_budget):
        srus = [{"position": (xs[b, i], ys[b, i]), "time": timestamps[i]} for i in range(active_sru)]
        batch.append(srus)
    
    sim_scores = sim.get_pos_batch(batch)
    max_idx = np.argmax(sim_scores)
    
    return sim_scores[max_idx], batch[max_idx]

def evaluate_hybrid(xs, ys, timestamps, active_sru, iteration_budget, sim, surrogate, top_k=50):
    """Filtre le batch avec le surrogate, puis évalue le Top K avec le simulateur."""
    neighbors_tensor = build_tensor_from_candidates(
        xs, ys, timestamps, surrogate.start_lon, surrogate.start_lat, surrogate.device
    )

    with torch.no_grad():
        pred = surrogate.model(
            surrogate.drift_tensor,
            neighbors_tensor,
            drift_tokens=surrogate.token
        )

    surrogate_scores = pred.squeeze(0)

    # Extraction du Top K
    k = min(top_k, iteration_budget)
    _, top_idx = torch.topk(surrogate_scores, k)
    top_idx = top_idx.cpu().numpy()

    # Reconstruction uniquement pour le Top K
    top_batch = []
    for idx in top_idx:
        srus = [{"position": (xs[idx, i], ys[idx, i]), "time": timestamps[i]} for i in range(active_sru)]
        top_batch.append(srus)

    # Évaluation finale par le simulateur
    sim_scores = sim.get_pos_batch(top_batch)
    local_max_idx = np.argmax(sim_scores)
    
    return sim_scores[local_max_idx], top_batch[local_max_idx]

def run_unified_ils(sim, bounds, active_sru, timestamps, time_budget, rng, eval_func, iteration_budget=50):

    start = time.time()
    center_x = (bounds[0] + bounds[1]) / 2
    center_y = (bounds[2] + bounds[3]) / 2
    
    best_solution = [{"position": (center_x, center_y), "time": t} for t in timestamps]
    best_score = sim.get_pos_batch([best_solution])[0]
    history = [(0.0, best_score)]

    while (time.time() - start) < time_budget:
        xs, ys = generate_candidates(best_solution, bounds, iteration_budget, active_sru, rng)
        
        best_candidate_score, best_candidate_sol = eval_func(xs, ys, iteration_budget)

        if best_candidate_score > best_score:
            best_score = best_candidate_score
            best_solution = best_candidate_sol

        elapsed = time.time() - start
        if elapsed < time_budget:
            history.append((elapsed, best_score))

    return history



def run_cma_es(sim, bounds, active_sru, timestamps, time_budget, rng, iteration_budget=50):


    start = time.time()

    dim = active_sru * 2

    # bounds -> format vectorisé
    lower = []
    upper = []
    for _ in range(active_sru):
        lower.extend([bounds[0], bounds[2]])
        upper.extend([bounds[1], bounds[3]])

    lower = np.array(lower)
    upper = np.array(upper)
    scale = upper - lower

    x0 = np.full(dim, 0.5)

    es = cma.CMAEvolutionStrategy(
        x0,
        0.2,
        {
            'seed': int(rng.randint(0, 1e6)),
            'bounds': [0.0, 1.0]
            #'popsize': iteration_budget
        }
    )

    best_score = -np.inf
    history = []

    while (time.time() - start) < time_budget:

        solutions = es.ask()

        real_solutions = lower + np.array(solutions) * scale

        batch = []
        for x in real_solutions:
            srus = [
                {"position": (x[2*i], x[2*i+1]), "time": timestamps[i]}
                for i in range(active_sru)
            ]
            batch.append(srus)

        scores = sim.get_pos_batch(batch)

        es.tell(solutions, [-s for s in scores])

        max_idx = np.argmax(scores)
        best_score = max(best_score, scores[max_idx])

        elapsed = time.time() - start
        if elapsed < time_budget:
            history.append((elapsed, best_score))

    return history


def load_drift_files(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npz")]
    files.sort()
    return files

def prewarm_surrogate(surrogate, active_sru, batch_size=10):
    
    dummy_xs = np.zeros((batch_size, active_sru), dtype=np.float32)
    dummy_ys = np.zeros((batch_size, active_sru), dtype=np.float32)
    dummy_timestamps = [0]*active_sru

    dummy_tensor = build_tensor_from_candidates(
        dummy_xs, dummy_ys, dummy_timestamps,
        surrogate.start_lon, surrogate.start_lat, surrogate.device
    )

    with torch.no_grad():
        _ = surrogate.model(
            surrogate.drift_tensor,
            dummy_tensor,
            drift_tokens=surrogate.token
        )
    
    torch.cuda.synchronize()
    print(f"Pre-warmup du surrogate terminé pour batch_size={batch_size}, active_sru={active_sru}")

def run_experiment_multi(drift_folder, runs=5, time_budget=1.0, active_sru=3):
    drift_files = load_drift_files(drift_folder)
    all_sim50 = []
    all_sim1000 = []
    all_hybrid = []
    all_cma = []
    i=0
    for d_idx, drift_path in enumerate(drift_files):
        print(f"\n=== Drift {d_idx+1}/{len(drift_files)} ===")
        sim = SimulatorLoader("")
        surrogate = SurrogateLoader("output/best_model.pt")

        sim.load_drift(drift_path)
        surrogate.load_drift(drift_path)

        bounds = compute_bounds(sim)

        for r in range(runs):
            print(f"  Run {r+1}/{runs}")
            rng = np.random.RandomState(i)
            i= i+1
            timestamps = [rng.uniform(6*3600, 40*3600) for _ in range(active_sru)]
            
            rng_state = rng.get_state()

            sim.create_srus([{"position": {"x":0, "y":0}, "time":0} for _ in range(active_sru)])
            surrogate.create_srus([{"position": {"x":0, "y":0}, "time":0} for _ in range(active_sru)])

            def eval_wrapper_classic(xs, ys, budget):
                return evaluate_classic(xs, ys, timestamps, active_sru, budget, sim)

            def eval_wrapper_hybrid(xs, ys, budget):
                return evaluate_hybrid(xs, ys, timestamps, active_sru, budget, sim, surrogate, top_k=50)

            rng.set_state(rng_state)
            sim_scores50 = run_unified_ils(sim, bounds, active_sru, timestamps, time_budget, rng, eval_wrapper_classic, iteration_budget=50)
            
            rng.set_state(rng_state)
            sim_scores1000 = run_unified_ils(sim, bounds, active_sru, timestamps, time_budget, rng, eval_wrapper_classic, iteration_budget=1000)

            #prewarm_surrogate(surrogate=surrogate, active_sru=active_sru, batch_size=1000)
            rng.set_state(rng_state)
            hybrid_scores = run_unified_ils(sim, bounds, active_sru, timestamps, time_budget, rng, eval_wrapper_hybrid, iteration_budget=1000)

            cma_scores = run_cma_es(sim, bounds, active_sru, timestamps, time_budget, rng)

            all_sim50.append(sim_scores50)
            all_sim1000.append(sim_scores1000)
            all_hybrid.append(hybrid_scores)
            all_cma.append(cma_scores)

    return all_sim50, all_sim1000, all_hybrid, all_cma

def plot_results_complet(sim_all, sim_all1000, hybrid_all, cma_all, folder, resolution=200):
    def mean_runs(runs):
        max_len = max(len(run) for run in runs)
        mean_times = []
        mean_scores = []
        for i in range(max_len):
            times_i = []
            scores_i = []
            for run in runs:
                if i < len(run):
                    t, v = run[i]
                    times_i.append(t)
                    scores_i.append(v)
            mean_times.append(np.mean(times_i))
            mean_scores.append(np.mean(scores_i))
        return np.array(mean_times), np.array(mean_scores)

    sim_mean_t, sim_mean_s = mean_runs(sim_all)
    sim1000_mean_t, sim1000_mean_s = mean_runs(sim_all1000)
    hybrid_mean_t, hybrid_mean_s = mean_runs(hybrid_all)
    cma_mean_t, cma_mean_s = mean_runs(cma_all)

    plt.figure(figsize=(10, 6))
    plt.plot(sim_mean_t, sim_mean_s, label="ILS Simulateur (50 voisins)")
    plt.plot(sim1000_mean_t, sim1000_mean_s, label="ILS Simulateur (1000 voisins)")
    plt.plot(hybrid_mean_t, hybrid_mean_s, label="ILS Hybride (1000 surrogate -> top50 sim)")
    plt.plot(cma_mean_t, cma_mean_s, label="CMA-ES (1000 surrogate -> top50 sim)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Best Score")
    plt.title("Convergence Moyenne (Index-based Mean)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(folder, "plot_1_mean_index.png"), dpi=300)

    all_runs = sim_all + sim_all1000 + hybrid_all
    max_time = max(run[-1][0] for run in all_runs)
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

    sim_interp = interpolate_runs(sim_all)
    sim1000_interp = interpolate_runs(sim_all1000)
    hybrid_interp = interpolate_runs(hybrid_all)
    cma_interp = interpolate_runs(cma_all)

    sim_mean_interp = sim_interp.mean(axis=0)
    sim1000_mean_interp = sim1000_interp.mean(axis=0)
    hybrid_mean_interp = hybrid_interp.mean(axis=0)
    cma_mean_interp = cma_interp.mean(axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(grid, sim_mean_interp, label="ILS Simulateur (50 voisins)")
    plt.plot(grid, sim1000_mean_interp, label="ILS Simulateur (1000 voisins)")
    plt.plot(grid, hybrid_mean_interp, label="ILS Hybride (1000 surrogate -> top50 sim)", linewidth=2)
    plt.plot(grid, cma_mean_interp, label="CMA-ES (1000 surrogate -> top50 sim)", linewidth=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Best Score")
    plt.title("Convergence Moyenne (Interpolated Time Grid)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(folder, "plot_2_mean_interpolate.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    plt.step(grid, sim_mean_interp, where='post', label="ILS Simulateur (50 voisins)")
    plt.step(grid, sim1000_mean_interp, where='post', label="ILS Simulateur (1000 voisins)")
    plt.step(grid, hybrid_mean_interp, where='post', label="ILS Hybride (1000 surrogate -> top50 sim)", linewidth=2)
    plt.step(grid, cma_mean_interp, where='post', label="CMA-ES (1000 surrogate -> top50 sim)", linewidth=2)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Best Score")
    plt.title("Convergence par Paliers Moyenne (Stepwise Plot)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(folder, "plot_3_stepwise_mean.png"), dpi=300)

    plt.show()

def create_experiment_folder_multi(runs, time_budget, active_sru, n_files):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"NOWARM_files{n_files}_runs{runs}_time{time_budget}_sru{active_sru}_{now}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def save_config(folder, drift_path, runs, time_budget, active_sru):
    config = {"drift_path": drift_path, "runs": runs, "time_budget": time_budget, "active_sru": active_sru}
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

def save_raw_results(folder, sim_all, sim_all1000, hybrid_all, cma_all):
    data = {"sim": sim_all, "sim1000": sim_all1000, "hybrid": hybrid_all, "cma_all": cma_all}
    with open(os.path.join(folder, "results.json"), "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    drift_folder = r"dataset\DriftReduced\validation"
    runs = 20
    time_budget = 3.0
    active_sru = 3

    drift_files = load_drift_files(drift_folder)
    folder = create_experiment_folder_multi(runs, time_budget, active_sru, len(drift_files))
    save_config(folder, drift_folder, runs, time_budget, active_sru)

    sim_all50, sim_all1000, hybrid_all, cma_all = run_experiment_multi(drift_folder, runs, time_budget, active_sru)
    save_raw_results(folder, sim_all50, sim_all1000, hybrid_all, cma_all)
    plot_results_complet(sim_all50, sim_all1000, hybrid_all, cma_all, folder)