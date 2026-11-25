import os
import pickle
from PyQt5.QtCore import QThread, pyqtSignal
import math
import time

class SitFileReaderThread(QThread):

    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._is_running = True

    def run(self):

        self._is_running = True
        
        if not os.path.exists(self.file_path):
            self.error.emit(f"Fichier introuvable: {self.file_path}")
            return

        try:
            with open(self.file_path, 'rb') as f:                
                file_content = pickle.load(f)
                
                self.progress_updated.emit(100)
                
                if self._is_running:
                    self.finished.emit(file_content)
                
        except Exception as e:
            self.error.emit(f"Erreur lors du chargement du fichier : {str(e)}")
    
    def stop(self):
        """
        Stop the thread execution
        """
        self._is_running = False