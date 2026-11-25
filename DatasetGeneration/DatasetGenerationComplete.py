import numpy as np
import json
from datetime import datetime, timedelta
from Utilities.Object import Object
from Utilities.SearchZone import SearchZone, vehicule
from Utilities.Position import Position
from scipy.stats import qmc
import random
import os
from Utilities.Object import *
from Utilities.SearchZone import SearchZone
from Utilities.Position import Position
import Simulator
from Controller.Enum.SimulatorType import *
from Repository.FunctionRepository import FunctionRepository, FunctionType
import sobol_seq
import threading
import queue


NOISE_X_INIT = 0.1
NOISE_Y_INIT = 0.1

NOISE_X_FINAL = 1.0
NOISE_Y_FINAL = 1.0



def load_drift_file(filepath, objectClassId="RAFT", objectClassName="RAFT", rad=100.0):

    data = np.load(filepath, allow_pickle=True)

    trajectory = data["trajectory"]        # (N, T, 2)
    start_lon = float(data["start_lon"])
    start_lat = float(data["start_lat"])
    start_timestamp_str = str(data["timestamp"])

    N, T, _ = trajectory.shape
    start_timestamp = datetime.fromisoformat(start_timestamp_str)
    initial_timestamp_int = int(start_timestamp.timestamp())

    # Création de l'objet
    obj = Object(
        numberOfScenarios=N,   # un Scenario par particule
        objectClassId=objectClassId,
        objectClassName=objectClassName,
        initialPosition=Position(start_lon, start_lat),
        initialTimestamp=initial_timestamp_int,
        rad=rad
    )
    obj.objectType = ObjectType.person
    timestep_seconds = 300  # 5 minutes

    # Ajout des particules pour chaque Scenario (chaque particule)
    for p in range(N):
        positions = trajectory[p]   # (T,2)
        for t in range(T):
            lon, lat = positions[t]
            particle_timestamp = initial_timestamp_int + t * timestep_seconds            
            obj.appendParticle(
                id=p,  # Scenario id = particule
                position=Position(float(lon), float(lat)),
                state=11,
                age=int(t * timestep_seconds/60),
                particle_timestamp=particle_timestamp,
                orientation=0
            )

    return obj, trajectory

def generate_sobol_noise(it, Z, X_init=0.1, Y_init=0.1, X_final=1, Y_final=1):

    # Facteurs de croissance
    factor_X = (X_final / X_init) ** (1 / Z)
    factor_Y = (Y_final / Y_init) ** (1 / Z)

    # Calcul des dimensions du rectangle courant
    X_curr = X_init * (factor_X ** it)
    Y_curr = Y_init * (factor_Y ** it)

    # --- Générer tous les points Sobol d'un coup ---
    sobol_points = sobol_seq.i4_sobol_generate(2, Z)  # shape = (Z, 2)
    sobol_point = sobol_points[it]                     # point correspondant à l'itération

    # Position centrée sur 0
    dx = -X_curr/2 + sobol_point[0] * X_curr
    dy = -Y_curr/2 + sobol_point[1] * Y_curr
    return dx, dy
def generate_dataset_single_drift(drift_file, output_file, simulator, max_sru=3, iterations_per_sru=1000):

    obj, trajectory = load_drift_file(drift_file)
    T = len(obj.scenarios[0].particles)
    N = len(obj.scenarios)

    simulator.addObject(obj)

    total_samples = max_sru * iterations_per_sru

    positions = np.zeros((total_samples, max_sru, 2), dtype=np.float32)
    times = np.zeros((total_samples, max_sru), dtype=np.int32)
    mask = np.zeros((total_samples, max_sru), dtype=np.float32)
    outputs = np.zeros(total_samples, dtype=np.float32)

    sample_index = 0

    path_template = None


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

        if path_template is None:
            path_template = [
                ((seg.startPoint.x, seg.startPoint.y),
                 (seg.endPoint.x, seg.endPoint.y),
                 seg.speed,
                 seg.pathType.name)
                for seg in sru.path
            ]

        sru_pool.append(sru)

    factor_X = (NOISE_X_FINAL / NOISE_X_INIT) ** (1 / iterations_per_sru)
    factor_Y = (NOISE_Y_FINAL / NOISE_Y_INIT) ** (1 / iterations_per_sru)


    for n_sru in range(1, max_sru + 1):

        print(f"=== Génération pour {n_sru} SRU ===")

        active_srus = sru_pool[:n_sru]

        simulator.addSru(active_srus[-1])

        sobol_gen = qmc.Sobol(d=n_sru, scramble=True)
        sobol_points = sobol_gen.random(iterations_per_sru)

        sobol_noise = sobol_seq.i4_sobol_generate(2 * max_sru, iterations_per_sru)

        particle_indices = np.random.randint(0, N, (iterations_per_sru, max_sru))

        for it in range(iterations_per_sru):
            for sru_idx, sru in enumerate(active_srus):
                sx = sobol_noise[it, 2*sru_idx]
                sy = sobol_noise[it, 2*sru_idx+1]

                X_curr = NOISE_X_INIT * (factor_X ** it)
                Y_curr = NOISE_Y_INIT * (factor_Y ** it)

                dx = -X_curr/2 + sx * X_curr
                dy = -Y_curr/2 + sy * Y_curr

                v_time = sobol_points[it, sru_idx]

                start_seconds = 2*3600 + v_time * (40*3600)
                start_index = min(int(start_seconds / 300), T-1)

                p_idx = particle_indices[it, sru_idx]

                lon, lat = trajectory[p_idx, start_index]

                sru.position.x = lon + dx
                sru.position.y = lat + dy

                sru.timestamp = int(obj.initialTimestamp + start_seconds)

            sru_infos = []
            for sru_idx, sru in enumerate(active_srus):
                v_time = sobol_points[it, sru_idx]
                start_seconds = 2*3600 + v_time * (40*3600)

                sru_infos.append({
                    "sru_idx": sru_idx,
                    "sru": sru,
                    "start_seconds": start_seconds
                })

            sru_infos_sorted = sorted(sru_infos, key=lambda x: x["start_seconds"])

            for sorted_idx, info in enumerate(sru_infos_sorted):
                sru = info["sru"]
                sru_idx = info["sru_idx"]  # indice original
                start_seconds = info["start_seconds"]

                positions[sample_index, sorted_idx, 0] = sru.position.x
                positions[sample_index, sorted_idx, 1] = sru.position.y
                times[sample_index, sorted_idx] = start_seconds
                mask[sample_index, sorted_idx] = 1.0

            simulator.editSrus(active_srus)

            sim_result = -1
            try_count = 0

            while sim_result == -1 and try_count < 3:
                sim_result = simulator.run()[0]
                try_count += 1

            if sim_result == -1:
                print("Simulation failed")
                return
            if sim_result < 0.1:
                sim_result = 0.0
            outputs[sample_index] = sim_result

            sample_index += 1

    for sru in sru_pool:
        simulator.removeSru(str(sru.id))

    simulator.removeObject(str(obj.id))

    np.savez_compressed(
        output_file,
        positions=positions,
        times=times,
        mask=mask,
        outputs=outputs,
        path_template=np.array(path_template, dtype=object),
        drift_file=drift_file,
        initial_timestamp=obj.initialTimestamp
    )


def worker(file_queue, output_dir, max_sru, iterations, simulator):

    while True:

        try:
            drift_file = file_queue.get(timeout=1)
        except queue.Empty:
            return

        try:

            name = os.path.splitext(os.path.basename(drift_file))[0]
            output_file = os.path.join(output_dir, name + ".dataset.npz")

            print(f"[THREAD {threading.get_ident()}] processing {drift_file}")

            generate_dataset_single_drift(
                drift_file,
                output_file,
                simulator,
                max_sru=max_sru,
                iterations_per_sru=iterations
            )
        except Exception as e:
            print("ERROR:", drift_file, e)

        finally:
            file_queue.task_done()

def fill_queue(folder, file_queue):

    for f in os.listdir(folder):

        if f.endswith(".npz"):

            file_queue.put(
                os.path.join(folder, f)
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

def batch_generate(folder, output_dir, num_threads, max_sru, iterations):

    os.makedirs(output_dir, exist_ok=True)

    file_queue = queue.Queue()

    fill_queue(folder, file_queue)

    threads = []

    for i in range(num_threads):

        simulator = create_simulator()   # IMPORTANT un simulateur par thread

        t = threading.Thread(
            target=worker,
            args=(file_queue, output_dir, max_sru, iterations, simulator),
            daemon=True
        )

        t.start()
        threads.append(t)

    print("Workers started")

    try:
        file_queue.join()

        # recharger nouveaux fichiers
        fill_queue(folder, file_queue)

    except KeyboardInterrupt:

        print("Stopping workers...")

def menu():
    folder = input("Dossier contenant les drifts : ")
    output_dir = input("Dossier output : ")

    threads = int(input("Nombre de threads : "))
    max_sru = int(input("Max SRU : "))
    iterations = int(input("Iterations par SRU : "))

    batch_generate(folder, output_dir, threads, max_sru, iterations)

if __name__ == "__main__":
    menu()