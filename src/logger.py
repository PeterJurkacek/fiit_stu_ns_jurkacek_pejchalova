from pathlib import Path
import tensorflow as tf
from src.utils import timestamp
from src import config
import logging
from tensorflow.keras.callbacks import CSVLogger, TensorBoard


# https://docs.python.org/2/howto/logging.html

class Logger:
    def __init__(self, log_name="log_name"):
        self.log_name = log_name
        self.log_id = f"{log_name}_{timestamp()}"
        path = Path('/labs').resolve()
        self.logs_dir = path / 'logs' / self.log_id
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = path / 'models' / self.log_id
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.logs_dir / 'runtime.log'
        self.logger = logging.getLogger()  # RESOLVED BUG https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
        self.fhandler = logging.FileHandler(filename=self.log_file_path, mode='a')

    def start(self):
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(asctime)s - %(message)s')
        self.fhandler.setFormatter(formatter)
        self.logger.addHandler(self.fhandler)
        self.logger.setLevel(logging.DEBUG)
        logging.info(f"Started: {self.log_id}")
        logging.info(f"log_id: {self.log_id}")
        logging.info(f"log_file_path: {self.log_file_path}")
        logging.info(f"logs_dir: {self.logs_dir}")
        logging.info(f"models_dir: {self.models_dir}")

    def end(self):
        logging.info(f"Finished: {self.log_id}")
        self.logger.removeHandler(self.fhandler)

    def get_callbacks(self, train_or_test: str, run_id: str):
        # logging.info(self.logs_dir / f'{train_or_test}_history_{run_id}.csv')
        return [
            TensorBoard(
                log_dir=self.logs_dir / f'{train_or_test}_tensorboard_{run_id}',
                histogram_freq=config.histogram_freq,
                profile_batch=config.profile_batch),
            CSVLogger(
                filename=str(self.logs_dir / f'{train_or_test}_history_{run_id}.csv'),
                append=True,
                separator=';')
        ]

    def get_model_path(self, run_id):
        model_path = self.models_dir / f"model_{run_id}.h5"
        return model_path
