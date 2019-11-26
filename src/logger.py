import shutil
from pathlib import Path
import tensorflow as tf
from src.utils import timestamp
from src import config
import logging
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorboard.plugins.hparams import api as hp

# https://docs.python.org/2/howto/logging.html

class Logger:
    def __init__(self, log_name="log_name"):
        self.log_name = log_name
        self.log_id = f"{log_name}"  # _{timestamp()}"
        path = Path('/labs').resolve()
        self.logs_dir = path / 'logs' / self.log_id
        shutil.rmtree(self.logs_dir, ignore_errors=True)
        self.logs_dir.mkdir(parents=True)
        self.models_dir = path / 'models' / self.log_id
        shutil.rmtree(self.models_dir, ignore_errors=True)
        self.models_dir.mkdir(parents=True)
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

    def get_callbacks(self, train_or_test: str, run_id: str, hparams):
        # logging.info(self.logs_dir / f'{train_or_test}_history_{run_id}.csv')
        return [
            TensorBoard(
                log_dir=str(self.logs_dir / run_id),
                histogram_freq=config.histogram_freq,
                profile_batch=config.profile_batch),
            CSVLogger(
                filename=str(self.logs_dir / run_id / f'history.csv'),
                append=True,
                separator=';'),
            # EarlyStopping(monitor='val_loss',
            #               min_delta=0,
            #               patience=3,
            #               verbose=1,
            #               restore_best_weights=True),
            # ModelCheckpoint(str(self.models_dir / f'{train_or_test}_model_checkpoint_{run_id}.h5'),
            #                 monitor='val_loss',
            #                 mode='min',
            #                 save_best_only=True,
            #                 verbose=1)
            hp.KerasCallback(str(self.logs_dir / run_id), hparams)
        ]

    def get_model_path(self, run_id):
        model_path = self.models_dir / f"model_{run_id}.h5"
        return model_path

    def setup_hparams_config(self):
        with tf.summary.create_file_writer(str(self.logs_dir)).as_default():
            hp.hparams_config(hparams=config.HPARAMS, metrics=config.METRICS)

