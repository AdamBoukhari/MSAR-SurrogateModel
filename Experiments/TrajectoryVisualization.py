#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_trajectory(filepath):
    data = np.load(filepath, allow_pickle=True)
    trajectory = data["trajectory"]  # (N, T, 2)
    return trajectory


def plot_filaments(trajectory, linewidth=0.5, alpha=0.3):
    N, T, _ = trajectory.shape

    plt.figure(figsize=(8, 8))

    for p in range(N):
        lon = trajectory[p, :, 0]
        lat = trajectory[p, :, 1]

        plt.plot(
            lon, lat,
            linewidth=linewidth,
            alpha=alpha
        )

    # point de départ
    plt.scatter(
        trajectory[:, 0, 0],
        trajectory[:, 0, 1],
        s=10,
        label="Start"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Particles")
    plt.axis("equal")

    plt.grid(False)
    plt.legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot drift trajectories from .npz file")

    parser.add_argument("--lw", type=float, default=0.5, help="Line width (default: 0.5)")
    parser.add_argument("--alpha", type=float, default=0.3, help="Transparency (default: 0.3)")

    args = parser.parse_args()
    path = input("Drift file path: ")

    trajectory = load_trajectory(path)
    plot_filaments(trajectory, linewidth=args.lw, alpha=args.alpha)


if __name__ == "__main__":
    while True:
        main()