import sys
from pathlib import Path
import tensorflow as tf

#Umozni pracovat s vlastnymi modulmi
sys.path.append('/labs')
from src.utils import timestamp


class Logger():
    def __init__(self):
        path = Path('/labs/logs')
        path = path.resolve()
        self.logs_dir = path / timestamp()
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.logs_dir,
                histogram_freq=1,
                profile_batch=0)
        ]
