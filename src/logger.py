import sys
from pathlib import Path
import tensorflow as tf

# Umozni pracovat s vlastnymi modulmi
from src.utils import timestamp
import logging


# https://docs.python.org/2/howto/logging.html

class Logger:
    def __init__(self, log_id=timestamp()):
        self.start_time = timestamp()
        self.log_id = log_id
        path = Path('/labs').resolve()
        print(f'Project path: {path}')
        self.logs_dir = path / 'logs' / self.log_id
        self.models_dir = path / 'models' / self.log_id
        self.log_file_path = self.logs_dir / 'runtime.log'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=self.log_file_path,
                            format='%(levelname)s: %(asctime)s=> %(message)s',
                            level=logging.DEBUG)

    def start(self):
        logging.info(f"Started: {self.log_id}")
        logging.info(f"Initialize Logger at: {self.start_time}")
        logging.info(f"log_id: {self.log_id}")
        logging.info(f"log_file_path: {self.log_file_path}")
        logging.info(f"logs_dir: {self.logs_dir}")
        logging.info(f"models_dir: {self.models_dir}")

    def end(self):
        logging.info(f"Finished: {self.log_id}")

    def get_callbacks(self, train_or_test: str, run_id: str):
        return [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.logs_dir / f'{train_or_test}_tensorboard_{run_id}',
                histogram_freq=1,
                profile_batch=0)
        ]

    def get_model_path(self, run_id):
        model_path = self.models_dir / f"model_{run_id}.h5"
        return model_path