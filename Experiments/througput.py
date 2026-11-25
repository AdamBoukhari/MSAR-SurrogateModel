import numpy as np
import time
from ModelLoader import SimulatorLoader, SurrogateLoader

MAX_SRU = 5

class ThroughputCLI:

    def __init__(self):

        self.sim_phys = SimulatorLoader("")
        self.sim_ml   = SurrogateLoader("output/best_model.pt")

        self.drift_loaded = False
        self.active_sru = 1
        self.batch_size = 1
        self.timestamps = [0]*MAX_SRU
        self.loader_choice = "surrogate"  # "surrogate" ou "simulator"

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

    def run(self):

        while True:

            print("\n===== THROUGHPUT MENU =====")
            print("1 - Load drift")
            print("2 - Configure SRU, batch size & loader")
            print("3 - Run continuous throughput test")
            print("0 - Exit")

            c = input("> ")

            if c == "1":
                self.menu_load_drift()
            elif c == "2":
                self.menu_sru()
            elif c == "3":
                self.run_stress_loop()
            elif c == "0":
                break

    def menu_load_drift(self):
        path = input("Drift file path: ")
        self.sim_phys.load_drift(path)
        self.sim_ml.load_drift(path)

        # initialize SRUs with zeros
        srus = [{"position":{"x":0,"y":0},"time":0} for _ in range(MAX_SRU)]
        self.sim_phys.create_srus(srus)
        self.sim_ml.create_srus(srus)

        self.drift_loaded = True
        print("Drift loaded")

    def menu_sru(self):
        self.active_sru = int(input(f"Number of active SRU (1-{MAX_SRU}): "))
        self.batch_size = int(input("Batch size: "))
        self.loader_choice = input("Loader to test (surrogate/simulator): ").strip().lower()

        srus = [{"position":{"x":0,"y":0},"time":0} for _ in range(self.active_sru)]
        self.sim_phys.create_srus(srus)

        for i in range(self.active_sru):
            t = float(input(f"Timestamp SRU {i}: "))
            self.timestamps[i] = t

    def generate_random_batch(self, bounds):
        min_x, max_x, min_y, max_y = bounds
        batch = []

        for _ in range(self.batch_size):
            srus = []
            for i in range(self.active_sru):
                px = np.random.uniform(min_x, max_x)
                py = np.random.uniform(min_y, max_y)
                srus.append({
                    "position": (px, py),
                    "time": self.timestamps[i]
                })
            batch.append(srus)

        return batch

    def run_stress_loop(self):

        if not self.drift_loaded:
            print("Load drift first")
            return

        bounds = self.compute_bounds()
        loader = self.sim_ml if self.loader_choice == "surrogate" else self.sim_phys

        total_samples = 0
        start_time = time.time()

        print(f"\nRunning stress test on {self.loader_choice} with batch size {self.batch_size}")
        print("Press Ctrl+C to stop...\n")

        try:
            while True:
                batch = self.generate_random_batch(bounds)
                # calcul POS en batch
                loader.get_pos_batch(batch)

                total_samples += len(batch)
                elapsed = time.time() - start_time
                throughput = total_samples / elapsed

                print(f"\rSamples processed: {total_samples}, Throughput: {throughput:.2f} POS/sec", end="")

        except KeyboardInterrupt:
            print("\nStress test stopped.")
            elapsed = time.time() - start_time
            throughput = total_samples / elapsed
            print(f"Total samples: {total_samples}, Average throughput: {throughput:.2f} POS/sec")


if __name__ == "__main__":
    cli = ThroughputCLI()
    cli.run()