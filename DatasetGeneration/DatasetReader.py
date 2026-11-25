import numpy as np
import os

def display_sample(file_path, index):

    if not os.path.exists(file_path):
        print("Fichier introuvable")
        return

    data = np.load(file_path, allow_pickle=True)

    positions = data["positions"]
    times = data["times"]
    mask = data["mask"]
    outputs = data["outputs"]
    path_template = data["path_template"]
    drift_file = data["drift_file"]

    n_samples = positions.shape[0]

    if index < 0 or index >= n_samples:
        print(f"Index invalide (0-{n_samples-1})")
        return

    print("\n================ SAMPLE ================")

    print("Drift file :", drift_file)

    print("\n--- INPUTS ---")

    print("\nPositions (lon, lat):")
    print(positions[index])

    print("\nTimes (seconds):")
    print(times[index])

    print("\nMask:")
    print(mask[index])

    print("\n--- OUTPUT ---")
    print(outputs[index])

    print("\n--- PATH TEMPLATE ---")
    print(path_template)

    print("========================================\n")


def main():

    try:
        while True:

            file_path = input("\nNom du fichier npz: ").strip()
            index_str = input("Index du sample (0-4999): ").strip()

            try:
                index = int(index_str)
            except:
                print("Index invalide")
                continue

            display_sample(file_path, index)

    except KeyboardInterrupt:
        print("\nFin du programme")


if __name__ == "__main__":
    main()