import numpy as np
import matplotlib.pyplot as plt

from ModelLoader import SimulatorLoader, SurrogateLoader

import cma
import nevergrad as ng
import optuna
import os
from scipy.optimize import differential_evolution
import json
from datetime import datetime

MAX_SRU = 5


class OptimizerCLI:

    def __init__(self):

        self.sim_phys = SimulatorLoader("")
        self.sim_ml   = SurrogateLoader("output/best_model.pt")

        self.drift_loaded = False

        self.compute_bounds_initialized = False

        self.active_sru = 1
        self.timestamps = [0]*MAX_SRU

        self.max_iter = 100
        self.optimizer = "cma"


        self.bounds = None

        self.history_surrogate = []
        self.history_simulator = []
        self.n_runs = 1


    def run(self):

        while True:

            print("\n===== OPTIMIZATION MENU =====")
            print("1 - Load drift")
            print("2 - Load drift folder") 
            print("3 - Configure SRU")
            print("4 - Optimizer settings")
            print("5 - Run optimization")
            print("0 - Exit")

            c = input("> ")

            if c == "1":
                self.menu_load_drift()
            elif c == "2":
                self.menu_load_drift_folder()

            elif c == "3":
                self.menu_sru()

            elif c == "4":
                self.menu_optimizer()

            elif c == "5":
                self.run_optimization()

            elif c == "0":
                break


    def menu_load_drift_folder(self):

        folder = input("Drift folder path: ")

        self.drift_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if  f.endswith(".npz") 
        ]

        if len(self.drift_files) == 0:
            print("No drift files found")
            return

        self.compute_bounds_initialized = False
        print(f"{len(self.drift_files)} drift files loaded")

    def initialize_srus(self):

        srus = []

        for _ in range(self.active_sru):

            srus.append({
                "position": {"x":0,"y":0},
                "time":0
            })

        self.sim_phys.create_srus(srus)
        self.sim_ml.create_srus(srus)


    def menu_sru(self):

        self.active_sru = int(input("Active SRU (1-5): "))
        #self.initialize_srus()
        for i in range(self.active_sru):

            t = float(input(f"Timestamp SRU {i}: "))
            self.timestamps[i] = t


    def menu_optimizer(self):

        print("\nChoose optimizer")

        print("1 - CMA-ES")
        print("2 - Nevergrad")
        print("3 - Differential Evolution")
        print("4 - Optuna")

        c = input("> ")

        if c == "1":
            self.optimizer = "cma"
        elif c == "2":
            self.optimizer = "nevergrad"
        elif c == "3":
            self.optimizer = "de"
        elif c == "4":
            self.optimizer = "optuna"

        self.max_iter = int(input("Iterations: "))
        self.n_runs = int(input("Number of runs (different seeds): "))


    def compute_bounds(self):
        if self.sim_phys.trajectory is None:
            print("No trajectory loaded")
            return
        traj = self.sim_phys.trajectory

        xs = traj[:,:,0]
        ys = traj[:,:,1]

        min_x = xs.min()
        max_x = xs.max()

        min_y = ys.min()
        max_y = ys.max()

        center_x = (min_x + max_x)/2
        center_y = (min_y + max_y)/2

        size = max(max_x-min_x, max_y-min_y)/2 + 0.5

        self.bounds = [
            (center_x-size, center_x+size),
            (center_y-size, center_y+size)
        ]


    def move_from_vector(self,x):

        updates = []

        for i in range(self.active_sru):

            px = x[2*i]
            py = x[2*i+1]

            updates.append({
                "id":i,
                "position":(px,py),
                "time":self.timestamps[i]
            })

        self.sim_ml.move_srus_batch(updates)
        self.sim_phys.move_srus_batch(updates)


    def surrogate_objective(self,x):

        self.move_from_vector(x)

        return -self.sim_ml.get_pos()

    def surrogate_objective_batch(self, X_batch):
        """
        X_batch: list of np.array (shape: dim)
        returns: list of scores
        """

        sru_batch = []

        for x in X_batch:
            srus = []

            for i in range(self.active_sru):
                srus.append({
                    "position": (float(x[2*i]), float(x[2*i+1])),
                    "time": self.timestamps[i]
                })

            sru_batch.append(srus)

        # appel batch réel au surrogate
        return [-v for v in self.sim_ml.get_pos_batch(sru_batch)]

    def simulator_objective(self,x):

        self.move_from_vector(x)

        return -self.sim_phys.get_pos()


    def evaluate_with_simulator(self,x):

        self.move_from_vector(x)

        return self.sim_phys.get_pos()


    def run_single_optimizer(self, objective, seed):

        dim = self.active_sru*2
        bounds = self.bounds*self.active_sru

        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        scale = upper - lower

        best_val = -np.inf
        history = []

        if self.optimizer == "cma":

            #x0 = np.array([(b[0]+b[1])/2 for b in bounds])
            x0 = np.full(dim, 0.5)
            es = cma.CMAEvolutionStrategy(x0,0.1,{'seed': seed,'bounds': [0.0, 1.0]  })

            for _ in range(self.max_iter):

                solutions = es.ask()
                real_solutions = lower + solutions * scale
                # convert list -> batch SRU
                if objective == self.surrogate_objective:
                    vals = self.surrogate_objective_batch(real_solutions)
                else:
                    # sinon fallback sur évaluation individuelle (simulateur)
                    vals = [objective(x) for x in real_solutions]
                # tell avec batch
                es.tell(solutions, vals)

                #best_x = es.result.xbest

                best_x = es.result.xbest
                best_x_real = lower + best_x * scale

                phys = self.evaluate_with_simulator(best_x_real)

                #phys = self.evaluate_with_simulator(best_x)

                best_val = max(best_val, phys)

                history.append(best_val)

        elif self.optimizer == "nevergrad":

            param = ng.p.Array(shape=(dim,))

            opt = ng.optimizers.NGOpt(parametrization=param,budget=self.max_iter)

            for _ in range(self.max_iter):

                x = opt.ask().value

                val = objective(x)

                opt.tell(x,val)

                phys = self.evaluate_with_simulator(x)

                best_val = max(best_val, phys)

                history.append(best_val)

        elif self.optimizer == "de":

            def callback(xk,convergence):

                nonlocal best_val

                phys = self.evaluate_with_simulator(xk)

                best_val = max(best_val, phys)

                history.append(best_val)

            differential_evolution(
                objective,
                bounds=bounds,
                maxiter=self.max_iter,
                callback=callback
            )

        elif self.optimizer == "optuna":

            def optuna_objective(trial):

                x=[]

                for i in range(self.active_sru):

                    bx,by=self.bounds

                    x.append(trial.suggest_float(f"x{i}",bx[0],bx[1]))
                    x.append(trial.suggest_float(f"y{i}",by[0],by[1]))

                val=objective(x)

                phys=self.evaluate_with_simulator(x)

                nonlocal best_val

                best_val=max(best_val,phys)

                history.append(best_val)

                return val

            study=optuna.create_study(direction="minimize")

            study.optimize(optuna_objective,n_trials=self.max_iter)

        return history
    
    def create_output_folder(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"results_{self.optimizer}_SRU{self.active_sru}_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)
        return folder_name
    
    def save_results(self, folder_name,
                    all_surrogate_global,
                    all_simulator_global,
                    all_final_tradeoffs,
                    all_negative_errors):

        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, list):
                return [to_serializable(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            else:
                return obj

        data = {
            "optimizer": self.optimizer,
            "active_sru": self.active_sru,
            "max_iter": self.max_iter,
            "n_runs": self.n_runs,
            #"bounds": self.bounds,

            "all_surrogate_global": all_surrogate_global,
            "all_simulator_global": all_simulator_global,
            "all_final_tradeoffs": all_final_tradeoffs,
            "all_negative_errors": all_negative_errors
        }
        path = os.path.join(folder_name, "results.json")

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Results saved to {path}")
    def run_optimization(self):

        if not hasattr(self, "drift_files"):
            print("Load drift folder first")
            return

        all_surrogate_global = []
        all_simulator_global = []
        all_final_tradeoffs = []
        all_negative_errors = []

        ran = 0
        for drift_idx, drift_path in enumerate(self.drift_files):

            print(f"\n===== DRIFT {drift_idx+1}/{len(self.drift_files)} =====")
            print(f"Loading: {drift_path}")

            self.sim_phys.load_drift(drift_path)
            self.sim_ml.load_drift(drift_path)

            self.compute_bounds()
            self.initialize_srus()

            for run in range(self.n_runs):

                print(f"\n--- RUN {run+1}/{self.n_runs} ---")
                ran+=1
                np.random.seed(ran)

                for i in range(self.active_sru):
                    self.timestamps[i] = np.random.uniform(2*3600, 40*3600)

                self.initialize_srus()

                print("Running surrogate-guided optimization...")
                hist_s = self.run_single_optimizer(self.surrogate_objective, run)
                print("Running simulator-only optimization...")
                hist_p = self.run_single_optimizer(self.simulator_objective, run)

                final_tradeoff = hist_p[-1] - hist_s[-1]  # positif = simulateur meilleur
                all_final_tradeoffs.append(final_tradeoff)

                # pour l'erreur négative moyenne (tradeoff < 0)
                error_neg = max(0, final_tradeoff)  # si surrogate meilleur -> positif, sinon 0
                all_negative_errors.append(error_neg)


                all_surrogate_global.append(hist_s)
                all_simulator_global.append(hist_p)
        folder_name = self.create_output_folder()
        self.save_results(
            folder_name,
            all_surrogate_global,
            all_simulator_global,
            all_final_tradeoffs,
            all_negative_errors
        )

        max_len = max(
            len(h) for h in all_surrogate_global + all_simulator_global
        )

        def pad(arr):
            return arr + [arr[-1]] * (max_len - len(arr))

        all_surrogate_global = np.array([pad(h) for h in all_surrogate_global])
        all_simulator_global = np.array([pad(h) for h in all_simulator_global])

        mean_s = all_surrogate_global.mean(axis=0)
        std_s  = all_surrogate_global.std(axis=0)

        mean_p = all_simulator_global.mean(axis=0)
        std_p  = all_simulator_global.std(axis=0)

        plt.figure(figsize=(8,5))

        plt.hist(all_final_tradeoffs, bins=150, density=True, alpha=0.7, edgecolor='black')
        plt.xlabel("Trade-off final (simulateur - surrogate)")
        plt.ylabel("Densité")
        plt.title("Distribution détaillée du trade-off final sur tous les runs")
        plt.xlim(-1, 1)  # ajuster selon la plage de tes données
        plt.show()
        # moyenne de l'erreur négative
        mean_negative_error = np.mean(all_negative_errors)
        print("Erreur négative moyenne :", mean_negative_error)

        x = np.arange(len(mean_s))

        plt.figure(figsize=(9,5))

        plt.plot(x, mean_s, label="Surrogate-guided")
        plt.fill_between(x, mean_s-std_s, mean_s+std_s, alpha=0.2)

        plt.plot(x, mean_p, label="Simulator-only")
        plt.fill_between(x, mean_p-std_p, mean_p+std_p, alpha=0.2)

        plt.xlabel("Iteration")
        plt.ylabel("Best POS (evaluated with simulator)")
        plt.title("GLOBAL convergence (all drifts + runs)")

        plt.legend()
        plt.show()

if __name__ == "__main__":

    cli = OptimizerCLI()

    cli.run()