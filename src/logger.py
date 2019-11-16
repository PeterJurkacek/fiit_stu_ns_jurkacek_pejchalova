import sys
from pathlib import Path
import tensorflow as tf

#Umozni pracovat s vlastnymi modulmi
sys.path.append('/labs')
from src.utils import timestamp


class Logger():
    def __init__(self):

        self.logs_dir = self.get_logs_dir()
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.logs_dir / timestamp(),
                histogram_freq=1,
                profile_batch=0)
        ]

    def get_logs_dir(self):
        path = Path('/labs/logs')
        return path.resolve()
