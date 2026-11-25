import re
from datetime import datetime
from Model.Scenario import Scenario
from Model.Particle import Particle
from Model.Object import Object
from Utilities.Position import Position
from PyQt5.QtCore import QThread, pyqtSignal
import time
import math
import numpy as np
from Model.Object import *
class DriftfileReaderThread(QThread):

    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(Object)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._is_running = True

    """ 
    def run(self):
        self._is_running = True
        total_lines = sum(1 for _ in open(self.file_path))
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        # Initialisation des variables
        object_class_id = object_class_name = None
        initial_position = None
        initial_timestamp = None
        particle_timestamp = None
        rad = None
        obj = None
        i = 0  # Pour suivre le nombre de scénarios
        while i < len(lines) and self._is_running:
            line = lines[i].strip()
            
            # Vérifie la section des informations de l'objet
            if line.startswith("# Object class id & name:"):
                i += 2
                self.progress_updated.emit(math.floor(((i/total_lines)*100)))
                object_class_id, object_class_name = lines[i].strip().split()
            
            # Vérifie la section des positions initiales
            elif line.startswith("# Seeding start time, position & radius:"):
                i += 2
                self.progress_updated.emit(math.floor(((i/total_lines)*100)))
                start_date, start_time, start_lon, start_lat, start_rad = lines[i].strip().split()
                timestamp = f"{start_date} {start_time}"
                initial_timestamp = int(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp())
                initial_position = Position(float(start_lon), float(start_lat))
                rad = float(start_rad)
            # Vérifie la section de la durée de la simulation
            elif line.startswith("# Total no of seeded particles:"):
                i += 2
                self.progress_updated.emit(math.floor(((i/total_lines)*100)))
                number_of_particles = int(lines[i].strip())
                number_of_scenarios = number_of_particles  # Chaque groupe de 5000

                # Crée l'objet
                obj = Object(number_of_scenarios, object_class_id, object_class_name, initial_position, initial_timestamp, rad)
            
            # Dates et temps
            elif line.startswith("nowDate"):
                i += 1
                self.progress_updated.emit(math.floor(((i/total_lines)*100)))
                now_date, now_time = lines[i].strip().split()
                timestamp = f"{now_date} {now_time}"

                particle_timestamp = int(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp())
            # Vérifie la section des données des particules
            elif line.startswith("# Particle data:"):
                i += 2  # Passer à la ligne des données de particules
                self.progress_updated.emit(math.floor(((i/total_lines)*100)))
                while i < len(lines) and not lines[i].strip().startswith("#"):
                    particle_data = list(map(str.strip, lines[i].strip().split()))
                    if len(particle_data) == 6:
                        id = int(particle_data[0]) - 1  # L'index doit commencer à 0
                        lon = float(particle_data[1])
                        lat = float(particle_data[2])
                        state = int(particle_data[3])
                        age = int(particle_data[4])
                        orientation = int(particle_data[5])
                        
                        # Ajoute la particule à l'objet
                        obj.appendParticle(id, Position(x=lon,y=lat), state, age, particle_timestamp, orientation)
                    i += 1  # Passer à la ligne suivante
                    self.progress_updated.emit(math.floor(((i/total_lines)*100)))
            i += 1  # Passer à la ligne suivante
            self.progress_updated.emit(math.floor(((i/total_lines)*100)))
        if self._is_running:
            self.finished.emit(obj)
        else:
            print("loading aborted")
    """    
    def run(self):
        self._is_running = True
        with open(self.file_path, 'r') as f:
            object_class_id = object_class_name = None
            initial_position = None
            initial_timestamp = None
            particle_timestamp = None
            rad = None
            obj = None
            line_count = 0
            last_progress_emit = 0

            for line in f:
                if not self._is_running:
                    break
                line = line.strip()
                line_count += 1
                if line_count - last_progress_emit > 2000:
                    self.progress_updated.emit(line_count)
                    last_progress_emit = line_count
                if line.startswith("# Object class id & name:"):
                    next(f); line_count += 1
                    object_class_id, object_class_name = next(f).split()
                    line_count += 1
                elif line.startswith("# Seeding start time"):
                    next(f); line_count += 1
                    parts = next(f).split()
                    line_count += 1

                    start_date, start_time = parts[0], parts[1]
                    start_lon, start_lat, start_rad = parts[2], parts[3], parts[4]

                    dt = datetime.fromisoformat(f"{start_date} {start_time}")
                    initial_timestamp = int(dt.timestamp())
                    initial_position = Position(float(start_lon), float(start_lat))
                    rad = float(start_rad)
                elif line.startswith("# Total no of seeded particles"):
                    next(f); line_count += 1
                    number_of_particles = int(next(f))
                    line_count += 1
                    obj = Object(number_of_particles,
                            object_class_id,
                            object_class_name,
                            initial_position,
                            initial_timestamp,
                            rad)
                elif line.startswith("nowDate"):
                    parts = next(f).split()
                    line_count += 1
                    dt = datetime.fromisoformat(f"{parts[0]} {parts[1]}")
                    particle_timestamp = int(dt.timestamp())

                elif line.startswith("# Particle data"):
                    next(f); line_count += 1

                    for pdata in f:
                        line_count += 1
                        pdata = pdata.strip()
                        if not pdata or pdata[0] == "#":
                            break

                        p = pdata.split()
                        id = int(p[0]) - 1

                        obj.appendParticle(id,Position(float(p[1]), float(p[2])),int(p[3]),int(p[4]),particle_timestamp,int(p[5]))

        if self._is_running:
            self.progress_updated.emit(100)
            self.finished.emit(obj)
            
    def stop(self):
        self._is_running = False



def load_npz_file(filepath, objectClassId="RAFT", objectClassName="RAFT", rad=100.0):
    """
    Charge un fichier .npz de drift et retourne un Object (particules et états temporels)
    Version vectorisée pour grande quantité de données
    """

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