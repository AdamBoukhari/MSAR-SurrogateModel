import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import Simulator
from Controller.Enum.SimulatorType import *
from Repository.FunctionRepository import FunctionRepository, FunctionType
from Utilities.Object import *
from Utilities.SearchZone import SearchZone
from Utilities.Position import Position
import numpy as np
from datetime import datetime, timedelta
import torch
from Model.model import SurrogateModel as TorchSurrogate
import copy
from concurrent.futures import ThreadPoolExecutor
SRU = Dict[str, Tuple[float, float] | float]


def load_drift_file(filepath, objectClassId="RAFT", objectClassName="RAFT", rad=100.0):


    data = np.load(filepath, allow_pickle=True)

    trajectory = data["trajectory"]        # (N, T, 2)
    start_lon = float(data["start_lon"])
    start_lat = float(data["start_lat"])
    start_timestamp_str = str(data["timestamp"])

    N, T, _ = trajectory.shape

    n_select = min(256, N)  # sécurité si N < 256
    indices = np.random.choice(N, size=n_select, replace=False)
    trajectory = trajectory[indices]

    N = trajectory.shape[0]

    start_timestamp = datetime.fromisoformat(start_timestamp_str)
    initial_timestamp_int = int(start_timestamp.timestamp())

    obj = Object(
        numberOfScenarios=N,   # un Scenario par particule sélectionnée
        objectClassId=objectClassId,
        objectClassName=objectClassName,
        initialPosition=Position(start_lon, start_lat),
        initialTimestamp=initial_timestamp_int,
        rad=rad
    )
    obj.objectType = ObjectType.person
    timestep_seconds = 300  # 5 minutes

    # Ajout des particules
    for p in range(N):
        positions = trajectory[p]   # (T,2)
        for t in range(T):
            lon, lat = positions[t]
            particle_timestamp = initial_timestamp_int + t * timestep_seconds            
            obj.appendParticle(
                id=p,
                position=Position(float(lon), float(lat)),
                state=11,
                age=int(t * timestep_seconds / 60),
                particle_timestamp=particle_timestamp,
                orientation=0
            )

    return obj, trajectory

# def load_drift_file(filepath, objectClassId="RAFT", objectClassName="RAFT", rad=100.0):
#     """
#     Charge un fichier .npz de drift et retourne un Object (particules et états temporels)
#     Version vectorisée pour grande quantité de données
#     """

#     data = np.load(filepath, allow_pickle=True)

#     trajectory = data["trajectory"]        # (N, T, 2)
#     start_lon = float(data["start_lon"])
#     start_lat = float(data["start_lat"])
#     start_timestamp_str = str(data["timestamp"])

#     N, T, _ = trajectory.shape
#     start_timestamp = datetime.fromisoformat(start_timestamp_str)
#     initial_timestamp_int = int(start_timestamp.timestamp())

#     # Création de l'objet
#     obj = Object(
#         numberOfScenarios=N,   # un Scenario par particule
#         objectClassId=objectClassId,
#         objectClassName=objectClassName,
#         initialPosition=Position(start_lon, start_lat),
#         initialTimestamp=initial_timestamp_int,
#         rad=rad
#     )
#     obj.objectType = ObjectType.person
#     timestep_seconds = 300  # 5 minutes

#     # Ajout des particules pour chaque Scenario (chaque particule)
#     for p in range(N):
#         positions = trajectory[p]   # (T,2)
#         for t in range(T):
#             lon, lat = positions[t]
#             particle_timestamp = initial_timestamp_int + t * timestep_seconds            
#             obj.appendParticle(
#                 id=p,  # Scenario id = particule
#                 position=Position(float(lon), float(lat)),
#                 state=11,
#                 age=int(t * timestep_seconds/60),
#                 particle_timestamp=particle_timestamp,
#                 orientation=0
#             )

#     return obj, trajectory


class BaseDriftModel(ABC):


    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.drift_data = None

        self._load_model(model_path)

    @abstractmethod
    def _load_model(self, model_path: str):
        pass

    @abstractmethod
    def load_drift(self, drift_file: str):
        pass

    @abstractmethod
    def create_srus(self, srus: List[SRU]):
        """
        Replace all existing SRUs with the provided list.
        """
        pass

    @abstractmethod
    def move_sru(self, sru_id: int, position: Tuple[float, float], time: float):
        pass

    @abstractmethod
    def move_srus_batch(self, updates: List[Dict]):
        """
        Update multiple SRUs at once.
        updates format:
        [
            {"id": int, "position": (x,y), "time": float},
            ...
        ]
        """
        pass

    @abstractmethod
    def get_pos(self, sru_id: int) -> Tuple[float, float]:
        pass

    @abstractmethod
    def get_pos_batch(self, list_srus: List[List[SRU]]) -> List[float]:
        """
        Compute POS for multiple SRU configurations.

        list_srus: List of SRU lists
            [
                [sru1, sru2, ...],   # config 1
                [sru1, sru2, ...],   # config 2
                ...
            ]

        Returns:
            List of POS (float)
        """
        pass


class SimulatorLoader(BaseDriftModel):
    """
    Physics simulator implementation.
    """

    def _load_model(self, model_path: str):
        self.simulator = self.create_simulator()
        
        self.obj = None
        self.trajectory = None
        self.srus = []

    def create_simulator(self):
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

    def load_drift(self, drift_file: str):
        if self.obj is not None:
            self.simulator.removeObject(str(self.obj.id))

        self.obj, self.trajectory = load_drift_file(drift_file)
        
        self.simulator.addObject(self.obj)

    def create_srus(self, srus: List[SRU]):
        if self.obj is None:
            raise RuntimeError("You must load drift before creating SRUs")
        
        for sru in self.srus:
            self.simulator.removeSru(str(sru.id))

        self.srus = []

        for i, s in enumerate(srus):
            sru = SearchZone(Position(0,0), timestamp=self.obj.initialTimestamp)

            sru.width = 35
            sru.height = 41
            sru.endeavorHour = 2
            sru.searchSpeed = 160
            sru.transitSpeed = 160
            sru.endeavor = sru.endeavorHour * sru.searchSpeed

            sru.generatePath()

            sru.position.x = s["position"]["x"]
            sru.position.y = s["position"]["y"]
            sru.timestamp = int(s["time"]+self.obj.initialTimestamp)

            self.simulator.addSru(sru)
            self.srus.append(sru)


    def move_sru(self, sru_id: int, position: Tuple[float, float], time: int):
        self.srus[sru_id].position.x = position[0]
        self.srus[sru_id].position.y = position[1]
        self.srus[sru_id].timestamp =int(time + self.obj.initialTimestamp)

        self.simulator.editSrus(self.srus)

    def move_srus_batch(self, updates: List[Dict]):
        for u in updates:
            sru_id = u["id"]
            x, y = u["position"]
            t = u["time"]

            self.srus[sru_id].position.x = x
            self.srus[sru_id].position.y = y
            self.srus[sru_id].timestamp = int(t + self.obj.initialTimestamp)

        self.simulator.editSrus(self.srus)

    def get_pos(self):
        return self.simulator.run()[0]

    def get_pos_batch(self, list_srus: List[List[SRU]]) -> List[float]:
        results = []

        if len(self.srus) == 0:
            raise ValueError("SRUs must be initialized before batch evaluation")

        n_sru_ref = len(self.srus)

        for srus in list_srus:

            if len(srus) != n_sru_ref:
                raise ValueError("All SRU configs must have the same number of SRUs")

            # update in-place
            for i, s in enumerate(srus):
                pos = s["position"]

                if isinstance(pos, dict):
                    x = float(pos["x"])
                    y = float(pos["y"])
                else:
                    x, y = map(float, pos)

                t = s["time"]

                self.srus[i].position.x = x
                self.srus[i].position.y = y
                self.srus[i].timestamp = int(t + self.obj.initialTimestamp)

            # UN seul call simulateur
            self.simulator.editSrus(self.srus)

            pos = self.get_pos()
            results.append(pos)

        return results

class SurrogateLoader(BaseDriftModel):


    def _load_model(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = TorchSurrogate(
            d_model=128,
            num_head=8#,
            #K=16,
            #N=256
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.srus = []

    def load_drift(self, drift_file):
        data = np.load(drift_file, allow_pickle=True)

        trajectory = data["trajectory"]
        self.start_lon = float(data["start_lon"])
        self.start_lat = float(data["start_lat"])

        N = trajectory.shape[0]

        drift_size = 256

        if N >= drift_size:
            idx = np.random.choice(N, drift_size, replace=False)
        else:
            idx = np.random.choice(N, drift_size, replace=True)

        traj = trajectory[idx].copy()

        traj[..., 0] -= self.start_lon
        traj[..., 1] -= self.start_lat

        T = traj.shape[1]

        if T < 577:
            last = traj[:, -1:, :]
            pad = np.repeat(last, 577 - T, axis=1)
            traj = np.concatenate([traj, pad], axis=1)

        self.drift_tensor = torch.from_numpy(traj).float().unsqueeze(0).to(self.device)

        self.token =self.model.encode_drift(self.drift_tensor)


    def create_srus(self, srus):
        self.srus = srus

    def move_sru(self, sru_id, position, time):
        self.srus[sru_id] = {
            "position": position,
            "time": time
        }

    def build_sru_tensor(self):
        n_sru = len(self.srus)
        pos = np.zeros((1, 1, n_sru, 5), dtype=np.float32)

        for i, s in enumerate(self.srus):

            position = s["position"]

            # Normalisation position -> (x, y)
            if isinstance(position, dict):
                x = float(position["x"])
                y = float(position["y"])
            else:
                x, y = map(float, position)
            
            t = s["time"]
            pos[0,0,i,0] = float(x) - self.start_lon
            pos[0,0,i,1] = float(y) - self.start_lat
            pos[0,0,i,2] = (t - (2*3600)) / (40*3600)
            pos[0,0,i,3] = y / 90.0
            pos[0,0,i,4] = 1

        return torch.from_numpy(pos).to(self.device)

    def build_sru_batch_tensor(self, list_srus: List[List[SRU]]):
        B = 1
        S = len(list_srus)
        max_sru = max(len(srus) for srus in list_srus)

        tensor = np.zeros((B, S, max_sru, 5), dtype=np.float32)

        for s_idx, srus in enumerate(list_srus):
            for i, s in enumerate(srus):

                position = s["position"]

                if isinstance(position, dict):
                    x = float(position["x"])
                    y = float(position["y"])
                else:
                    x, y = map(float, position)

                t = s["time"]

                tensor[0, s_idx, i, 0] = x - self.start_lon
                tensor[0, s_idx, i, 1] = y - self.start_lat
                tensor[0, s_idx, i, 2] = (t - (2*3600)) / (40*3600)
                tensor[0, s_idx, i, 3] = y / 90.0
                tensor[0, s_idx, i, 4] = 1  # mask actif

        return torch.from_numpy(tensor).to(self.device)

    def move_srus_batch(self, updates: List[Dict]):
        for u in updates:
            sru_id = u["id"]

            self.srus[sru_id] = {
                "position": u["position"],
                "time": u["time"]
            }

    def get_pos(self):
        with torch.no_grad():

            sru_tensor = self.build_sru_tensor()

            pred = self.model(self.drift_tensor, sru_tensor, drift_tokens=self.token)

            return float(pred.item())
    
    def get_pos_batch(self, list_srus: List[List[SRU]]) -> List[float]:
        with torch.no_grad():

            sru_tensor = self.build_sru_batch_tensor(list_srus)

            pred = self.model(
                self.drift_tensor,
                sru_tensor,
                drift_tokens=self.token
            )  # shape: (1, S)

            return pred.squeeze(0).cpu().tolist()