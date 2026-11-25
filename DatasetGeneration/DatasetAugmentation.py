import numpy as np
import os
import queue
import threading
from datetime import datetime
from scipy.stats import qmc

from Utilities.Object import Object, ObjectType
from Utilities.SearchZone import SearchZone
from Utilities.Position import Position
import Simulator
from Controller.Enum.SimulatorType import SimulatorType
from Repository.FunctionRepository import FunctionRepository, FunctionType


def load_drift_file(filepath, objectClassId="RAFT", objectClassName="RAFT", rad=100.0):

    data = np.load(filepath, allow_pickle=True)

    trajectory = data["trajectory"]        # (N, T, 2)
    start_lon = float(data["start_lon"])
    start_lat = float(data["start_lat"])
    start_timestamp_str = str(data["timestamp"])

    N, T, _ = trajectory.shape
    start_timestamp = datetime.fromisoformat(start_timestamp_str)
    initial_timestamp_int = int(start_timestamp.timestamp())

    obj = Object(
        numberOfScenarios=N,
        objectClassId=objectClassId,
        objectClassName=objectClassName,
        initialPosition=Position(start_lon, start_lat),
        initialTimestamp=initial_timestamp_int,
        rad=rad
    )
    obj.objectType = ObjectType.person
    timestep_seconds = 300  # 5 minutes

    for p in range(N):
        positions = trajectory[p]
        for t in range(T):
            lon, lat = positions[t]
            particle_timestamp = initial_timestamp_int + t * timestep_seconds            
            obj.appendParticle(
                id=p,
                position=Position(float(lon), float(lat)),
                state=11,
                age=int(t * timestep_seconds/60),
                particle_timestamp=particle_timestamp,
                orientation=0
            )

    return obj, trajectory

def compute_bounds(trajectory):

    xs = trajectory[:, :, 0]
    ys = trajectory[:, :, 1]

    min_x = xs.min()
    max_x = xs.max()
    min_y = ys.min()
    max_y = ys.max()

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    size = max(max_x - min_x, max_y - min_y) / 2 + 0.5

    return (
        center_x - size, # min_x
        center_x + size, # max_x
        center_y - size, # min_y
        center_y + size  # max_y
    )

def maybe_break():
    pass

def augment_dataset_single_drift(drift_file, dataset_file, output_file, simulator, max_sru=5, new_iterations_per_sru=500):
    
    obj, trajectory = load_drift_file(drift_file)
    simulator.addObject(obj)

    bound_min_x, bound_max_x, bound_min_y, bound_max_y = compute_bounds(trajectory)

    old_data = np.load(dataset_file, allow_pickle=True)
    old_positions = old_data["positions"]
    old_times = old_data["times"]
    old_mask = old_data["mask"]
    old_outputs = old_data["outputs"]
    path_template = old_data["path_template"]

    total_new_samples = max_sru * new_iterations_per_sru
    new_positions = np.zeros((total_new_samples, max_sru, 2), dtype=np.float32)
    new_times = np.zeros((total_new_samples, max_sru), dtype=np.int32)
    new_mask = np.zeros((total_new_samples, max_sru), dtype=np.float32)
    new_outputs = np.zeros(total_new_samples, dtype=np.float32)

    sample_index = 0

    sru_pool = []
    for i in range(max_sru):
        sru = SearchZone(Position(0,0), timestamp=obj.initialTimestamp)
        sru.width = 35
        sru.height = 41
        sru.endeavorHour = 2
        sru.searchSpeed = 160
        sru.transitSpeed = 160
        sru.endeavor = sru.endeavorHour * sru.searchSpeed
        sru.generatePath()
        sru_pool.append(sru)

    for n_sru in range(1, max_sru + 1):
        active_srus = sru_pool[:n_sru]
        simulator.addSru(active_srus[-1])

        # Sobol global pour [X, Y, Temps] pour chaque SRU actif
        # d = 3 dimensions * le nombre de SRU
        sobol_gen = qmc.Sobol(d=3 * n_sru, scramble=True)
        sobol_points = sobol_gen.random(new_iterations_per_sru)

        for it in range(new_iterations_per_sru):
            sru_infos = []

            for sru_idx, sru in enumerate(active_srus):
                # Récupération des points Sobol mappés pour ce SRU précis
                sx = sobol_points[it, 3 * sru_idx]
                sy = sobol_points[it, 3 * sru_idx + 1]
                st = sobol_points[it, 3 * sru_idx + 2]

                # Mapping spatial (Global)
                sru.position.x = bound_min_x + sx * (bound_max_x - bound_min_x)
                sru.position.y = bound_min_y + sy * (bound_max_y - bound_min_y)

                start_seconds = 2 * 3600 + st * (40 * 3600)
                sru.timestamp = int(obj.initialTimestamp + start_seconds)

                sru_infos.append({
                    "sru_idx": sru_idx,
                    "sru": sru,
                    "start_seconds": start_seconds
                })

            # Trier temporellement pour le dataset
            sru_infos_sorted = sorted(sru_infos, key=lambda x: x["start_seconds"])

            for sorted_idx, info in enumerate(sru_infos_sorted):
                sru = info["sru"]
                new_positions[sample_index, sorted_idx, 0] = sru.position.x
                new_positions[sample_index, sorted_idx, 1] = sru.position.y
                new_times[sample_index, sorted_idx] = info["start_seconds"]
                new_mask[sample_index, sorted_idx] = 1.0

            # Simulation
            simulator.editSrus(active_srus)
            sim_result = -1
            try_count = 0

            while sim_result == -1 and try_count < 3:
                sim_result = simulator.run()[0]
                try_count += 1

            if sim_result == -1:
                print("Simulation failed for one sample, skipping...")
                sim_result = 0.0
            elif sim_result < 0.1:
                sim_result = 0.0
            elif sim_result > 0.1:
                maybe_break()
            
            
            new_outputs[sample_index] = sim_result
            sample_index += 1

    # Nettoyage Simulateur
    for sru in sru_pool:
        simulator.removeSru(str(sru.id))
    simulator.removeObject(str(obj.id))

    final_positions = np.concatenate((old_positions, new_positions), axis=0)
    final_times = np.concatenate((old_times, new_times), axis=0)
    final_mask = np.concatenate((old_mask, new_mask), axis=0)
    final_outputs = np.concatenate((old_outputs, new_outputs), axis=0)

    np.savez_compressed(
        output_file,
        positions=final_positions,
        times=final_times,
        mask=final_mask,
        outputs=final_outputs,
        path_template=path_template,
        drift_file=drift_file,
        initial_timestamp=obj.initialTimestamp
    )

def create_simulator():
    funRep = FunctionRepository()
    original_name = funRep.functions[0].name

    funRep.updateFunction(
        originalName=original_name,
        name="LRC",
        type=FunctionType.gaussian,
        height=0.8,
        threshold=2.82
    )

    simulator = Simulator.Simulator()
    simulator.changeSimulatorType(SimulatorType.GPUCalculatorLegTotal)
    simulator.updateFunctions(funRep.functions_indexes_py, funRep.functions)
    return simulator

def worker(file_queue, dataset_dir, output_dir, max_sru, iterations, simulator):
    while True:
        try:
            drift_file = file_queue.get(timeout=1)
        except queue.Empty:
            return

        try:
            name = os.path.splitext(os.path.basename(drift_file))[0]
            dataset_file = os.path.join(dataset_dir, name + ".dataset.npz")
            output_file = os.path.join(output_dir, name + ".dataset.npz")

            if not os.path.exists(dataset_file):
                print(f"[THREAD {threading.get_ident()}] Ancien dataset manquant pour {drift_file}, skip...")
                continue

            print(f"[THREAD {threading.get_ident()}] processing {drift_file}")

            augment_dataset_single_drift(
                drift_file,
                dataset_file,
                output_file,
                simulator,
                max_sru=max_sru,
                new_iterations_per_sru=iterations
            )
        except Exception as e:
            print(f"ERROR: {drift_file} -> {e}")
        finally:
            file_queue.task_done()

def fill_queue(folder, file_queue):
    for f in os.listdir(folder):
        if f.endswith(".npz") and not f.endswith(".dataset.npz"):
            file_queue.put(os.path.join(folder, f))

def batch_augment(drift_dir, dataset_dir, output_dir, num_threads, max_sru, iterations):
    os.makedirs(output_dir, exist_ok=True)
    file_queue = queue.Queue()
    fill_queue(drift_dir, file_queue)

    threads = []
    for i in range(num_threads):
        simulator = create_simulator()
        t = threading.Thread(
            target=worker,
            args=(file_queue, dataset_dir, output_dir, max_sru, iterations, simulator),
            daemon=True
        )
        t.start()
        threads.append(t)

    print("Workers started pour l'augmentation...")
    try:
        file_queue.join()
        print("Data Augmentation terminée avec succès.")
    except KeyboardInterrupt:
        print("Stopping workers...")

def menu():
    drift_dir = input("1. Dossier contenant les drifts originaux (.npz) : ")
    dataset_dir = input("2. Dossier contenant les anciens datasets (.dataset.npz) : ")
    output_dir = input("3. Dossier de sortie pour les datasets augmentés : ")

    threads = int(input("Nombre de threads : "))
    max_sru = int(input("Max SRU (ex: 3) : "))
    iterations = int(input("Nouvelles itérations par SRU à rajouter (ex: 500) : "))

    batch_augment(drift_dir, dataset_dir, output_dir, threads, max_sru, iterations)

if __name__ == "__main__":
    menu()