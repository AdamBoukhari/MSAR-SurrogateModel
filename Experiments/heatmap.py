import numpy as np
import matplotlib.pyplot as plt
from ModelLoader import SimulatorLoader, SurrogateLoader
from tqdm import tqdm

class HeatmapCLI:

    def __init__(self):
        self.sim_phys = SimulatorLoader("")
        self.sim_ml   = SurrogateLoader("output/best_model.pt")
        
        self.drift_loaded = False
        self.timestamp = 0
        self.resolution = 50
        self.fixed_srus = []


    def run(self):

        while True:

            print("\n===== MAIN MENU =====")
            print("1 - Load drift")
            print("2 - Configure SRU")
            print("3 - Heatmap settings")
            print("4 - Run heatmap")
            print("0 - Exit")

            choice = input("> ")

            if choice == "1":
                self.menu_load_drift()

            elif choice == "2":
                self.menu_sru()

            elif choice == "3":
                self.menu_heatmap_settings()

            elif choice == "4":
                self.menu_run_heatmap()

            elif choice == "0":
                break


    def menu_load_drift(self):

        while True:

            print("\n--- LOAD DRIFT ---")
            print("1 - Load drift file")
            print("0 - Back")

            c = input("> ")

            if c == "1":

                path = input("Drift file path: ")

                try:
                    self.sim_phys.load_drift(path)
                    self.sim_ml.load_drift(path)
                    self.drift_loaded = True
                    print("Drift loaded")

                except Exception as e:
                    print("Error:", e)

            elif c == "0":
                return


    def menu_sru(self):

        while True:

            print("\n--- SRU MENU ---")
            print("1 - Set fixed SRU")
            print("2 - Clear SRU")
            print("0 - Back")

            c = input("> ")

            if c == "1":

                n = int(input("How many fixed SRU: "))

                self.fixed_srus = []

                for i in range(n):

                    x = float(input(f"SRU {i} x: "))
                    y = float(input(f"SRU {i} y: "))
                    t = float(input(f"SRU {i} time: "))

                    self.fixed_srus.append({
                        "position": {"x": x, "y": y},
                        "time": t
                    })

                print("SRU updated")

            elif c == "2":

                self.fixed_srus = []
                print("SRU cleared")

            elif c == "0":
                return


    def menu_heatmap_settings(self):

        while True:

            print("\n--- HEATMAP SETTINGS ---")
            print("1 - Set timestamp")
            print("2 - Set resolution")
            print("0 - Back")

            c = input("> ")

            if c == "1":

                self.timestamp = float(input("Timestamp: "))

            elif c == "2":

                self.resolution = int(input("Resolution: "))

            elif c == "0":
                return


    def compute_bounds(self):

        traj = self.sim_phys.trajectory

        xs = traj[:, :, 0]
        ys = traj[:, :, 1]

        min_x = xs.min()
        max_x = xs.max()

        min_y = ys.min()
        max_y = ys.max()

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        size = max(max_x - min_x, max_y - min_y) / 2 + 0.5

        return (
            center_x - size,
            center_x + size,
            center_y - size,
            center_y + size
        )


    def menu_run_heatmap(self):

        if not self.drift_loaded:
            print("Load drift first")
            return

        while True:

            print("\n--- HEATMAP ---")
            print("1 - Run heatmap")
            print("0 - Back")

            c = input("> ")

            if c == "1":

                srus = self.fixed_srus.copy()

                srus.append({
                    "position": {"x": 0, "y": 0},
                    "time": 0
                })

                self.sim_phys.create_srus(srus)
                self.sim_ml.create_srus(srus)

                min_x, max_x, min_y, max_y = self.compute_bounds()

                xs = np.linspace(min_x, max_x, self.resolution)
                ys = np.linspace(min_y, max_y, self.resolution)

                heatmap_phys = np.zeros((self.resolution, self.resolution))
                heatmap_ml   = np.zeros((self.resolution, self.resolution))

                movable = len(self.sim_phys.srus) - 1

                total = self.resolution * self.resolution

                # build toutes les configs
                batch_size = 1024

                total = self.resolution * self.resolution

                with tqdm(total=total, desc="Running batch simulations") as pbar:

                    batch = []
                    indices = []

                    for i, x in enumerate(xs):
                        for j, y in enumerate(ys):

                            srus_config = self.fixed_srus.copy()

                            srus_config.append({
                                "position": {"x": x, "y": y},
                                "time": self.timestamp
                            })

                            batch.append(srus_config)
                            indices.append((i, j))

                            if len(batch) == batch_size:

                                phys = self.sim_phys.get_pos_batch(batch)
                                ml   = self.sim_ml.get_pos_batch(batch)

                                for (ii, jj), p, m in zip(indices, phys, ml):
                                    heatmap_phys[jj, ii] = p
                                    heatmap_ml[jj, ii]   = m

                                pbar.update(len(batch))

                                batch = []
                                indices = []

                    if batch:
                        phys = self.sim_phys.get_pos_batch(batch)
                        ml   = self.sim_ml.get_pos_batch(batch)

                        for (ii, jj), p, m in zip(indices, phys, ml):
                            heatmap_phys[jj, ii] = p
                            heatmap_ml[jj, ii]   = m

                        pbar.update(len(batch))

                fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.5))

                vmin = 0.0
                vmax = 1.0
                norm = None

                im0 = ax[0].imshow(
                    heatmap_phys,
                    extent=[min_x, max_x, min_y, max_y],
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="inferno",
                    norm=norm
                )
                ax[0].set_title("Search Simulator", fontsize=19)

                im1 = ax[1].imshow(
                    heatmap_ml,
                    extent=[min_x, max_x, min_y, max_y],
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="inferno",
                    norm=norm
                )
                ax[1].set_title("Surrogate Model", fontsize=19)

                for a in ax:
                    a.tick_params(axis='both', labelsize=15)

                #cbar = fig.colorbar(im0, ax=ax.ravel().tolist(), fraction=0.025, pad=0.04)
                #cbar.ax.tick_params(labelsize=12)
                #cbar.set_label("Score", fontsize=14)

                plt.tight_layout()
                plt.show()
            elif c == "0":
                return


if __name__ == "__main__":

    cli = HeatmapCLI()
    cli.run()