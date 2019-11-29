import shutil

import absl
import tensorflow as tf
from absl import logging
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorboard.plugins.hparams import api as hp

class Logger:
    def __init__(self, logs_dir,
                 models_dir,
                 hyperparams,
                 histogram_freq,
                 profile_batch,
                 update_freq):
        self.logs_dir = logs_dir
        self.models_dir = models_dir
        self.hparams = hyperparams
        self.histogram_freq = histogram_freq
        self.profile_batch = profile_batch
        self.update_freq = update_freq
        shutil.rmtree(self.logs_dir, ignore_errors=True)
        shutil.rmtree(self.models_dir, ignore_errors=True)
        self.logs_dir.mkdir(parents=True)
        self.models_dir.mkdir(parents=True)
        logging.get_absl_handler().use_absl_log_file('runtime.log', self.logs_dir)
        absl.flags.FLAGS.mark_as_parsed()
        logging.set_verbosity(logging.INFO)
        self.log_file_path = f"file://{logging.get_log_file_name()}"

    def start(self):
        if self.hparams is not None:
            with tf.summary.create_file_writer(str(self.logs_dir)).as_default():
                hp.hparams_config(hparams=self.hparams.HPARAMS, metrics=self.hparams.METRICS)
        logging.info(f"LOGGING START. Data are saving to logs_dir: {self.logs_dir}, models_dir: {self.models_dir}")
        logging.info(f"log_file_name:{self.log_file_path}")

    def end(self):
        logging.info(f"LOGGING END. Data was saved to logs_dir: {self.logs_dir}, models_dir: {self.models_dir}")

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

    def create_scalar_summary(self, run_id, hparams, accuracy):
        with tf.summary.create_file_writer(str(self.logs_dir / run_id)).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            tf.summary.scalar(name='Accuracy', data=accuracy, step=int(run_id))
