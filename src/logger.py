import sys, os
from pathlib import Path
import tensorflow as tf

#Umozni pracovat s vlastnymi modulmi
sys.path.append('/labs')
from src.utils import timestamp
from src import config

class Logger():
    def __init__(self, log_id=timestamp()):
        self.log_id = log_id
        path = Path('/labs').resolve()
        self.logs_dir = path / 'logs' / self.log_id
        print(self.logs_dir)
        self.model_path = path / 'models' / f"{self.log_id}_model.h5"
        print(self.model_path)
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.logs_dir,
                histogram_freq=config.histogram_freq,
                profile_batch=config.profile_batch)
        ]
        self.logs = []

    def get_model_path(self):
        return self.model_path

    def print(self, log: str):
        self.logs.append(log)

    def close(self):
        # print(f"TODO: Zapíš do súboru")
        f = open(os.path.join(self.logs_dir, self.log_id), "w+")
        f.write(self.logs),
        f.close()
        # self.logs zapíš do súboru