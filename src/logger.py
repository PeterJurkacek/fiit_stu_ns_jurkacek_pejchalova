import shutil
import tensorflow as tf
import logging
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorboard.plugins.hparams import api as hp

# https://docs.python.org/2/howto/logging.html
from src.config import Config


class Logger:
    def __init__(self, config: Config):
        self.logs_dir = config.logs_dir
        self.models_dir = config.models_dir
        shutil.rmtree(self.logs_dir, ignore_errors=True)
        shutil.rmtree(self.models_dir, ignore_errors=True)
        self.logs_dir.mkdir(parents=True)
        self.models_dir.mkdir(parents=True)
        self.metrics = config.metrics
        self.logger = logging.getLogger()  # RESOLVED BUG https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
        self.log_file_path = self.logs_dir / 'runtime.log'
        self.fhandler = logging.FileHandler(filename=self.log_file_path, mode='a')
        self.hparams = config.hyperparams
        self.histogram_freq = config.histogram_freq
        self.profile_batch = config.profile_batch

    def start(self):
        if self.hparams is not None:
            with tf.summary.create_file_writer(str(self.logs_dir)).as_default():
                hp.hparams_config(hparams=self.hparams.HPARAMS, metrics=self.hparams.METRICS)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(asctime)s - %(message)s')
        self.fhandler.setFormatter(formatter)
        self.logger.addHandler(self.fhandler)
        self.logger.setLevel(logging.DEBUG)
        logging.info(f"LOGGING START. Data are saving to logs_dir: {self.logs_dir}, models_dir: {self.models_dir}")

    def end(self):
        logging.info(f"LOGGING END. Data was saved to logs_dir: {self.logs_dir}, models_dir: {self.models_dir}")
        self.logger.removeHandler(self.fhandler)

    def get_model_path(self, run_id):
        model_path = self.models_dir / f"model_{run_id}.h5"
        return model_path

    def create_tensorboard_callback(self, run_id):
        return TensorBoard(
            log_dir=str(self.logs_dir / run_id),
            histogram_freq=self.histogram_freq,
            profile_batch=self.profile_batch)

    def create_csv_logger_callback(self, run_id):
        return CSVLogger(
            filename=str(self.logs_dir / run_id / f'history.csv'),
            append=True,
            separator=';')

    def create_hp_params_callback(self, run_id, hparams):
        return hp.KerasCallback(str(self.logs_dir / run_id), hparams)

    def create_early_stopping_callback(self):
        return EarlyStopping(monitor='val_loss',
                             min_delta=0,
                             patience=3,
                             verbose=1,
                             restore_best_weights=True),

    def create_model_checkpoint_callback(self, run_id):
        return ModelCheckpoint(str(self.models_dir / f'm{run_id}_checkpoint.h5'),
                               monitor='val_loss',
                               mode='min',
                               save_best_only=True,
                               verbose=1)
