import numpy as np
from pathlib import Path


def reduce_particles_in_file(input_path, output_path, target_N=500):
    data = np.load(input_path, allow_pickle=True)

    trajectory = data["trajectory"]  # (N, T, 2)
    N = trajectory.shape[0]

    if N <= target_N:
        print(f"Skipping {input_path.name} (N={N} <= {target_N})")
        trajectory_reduced = trajectory
    else:
        idx = np.random.choice(N, target_N, replace=False)
        trajectory_reduced = trajectory[idx]

    save_dict = {}
    for key in data.files:
        if key == "trajectory":
            save_dict[key] = trajectory_reduced
        else:
            save_dict[key] = data[key]

    np.savez(output_path, **save_dict)


def process_folder(input_dir, output_dir, target_N=500):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.npz"))

    print(f"{len(files)} fichiers trouvés")

    for file in files:
        output_file = output_dir / file.name
        print(f"Processing {file.name}")
        reduce_particles_in_file(file, output_file, target_N)


if __name__ == "__main__":

    input_dir = input("Dossier input contenant les .npz : ").strip()
    output_dir = input("Dossier output pour sauvegarder les fichiers réduits : ").strip()

    process_folder(input_dir, output_dir, target_N=500)

    print("Terminé.")