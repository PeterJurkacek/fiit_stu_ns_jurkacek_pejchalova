import shutil
import tensorflow as tf
from src import config
import logging
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorboard.plugins.hparams import api as hp

# https://docs.python.org/2/howto/logging.html
logs_dir = config.logs_dir
models_dir = config.models_dir
shutil.rmtree(logs_dir, ignore_errors=True)
shutil.rmtree(models_dir, ignore_errors=True)
logs_dir.mkdir(parents=True)
models_dir.mkdir(parents=True)
logger = logging.getLogger()  # RESOLVED BUG https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
log_file_path = logs_dir / 'runtime.log'
fhandler = logging.FileHandler(filename=log_file_path, mode='a')


def start():
    with tf.summary.create_file_writer(str(logs_dir)).as_default():
        hp.hparams_config(hparams=config.HPARAMS, metrics=config.METRICS)
    formatter = logging.Formatter('%(levelname)s - %(name)s - %(asctime)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
    logging.info(f"LOGGING START. Data are saving to logs_dir: {logs_dir}, models_dir: {models_dir}")


def end():
    logging.info(f"LOGGING END. Data was saved to logs_dir: {logs_dir}, models_dir: {models_dir}")
    logger.removeHandler(fhandler)


def get_model_path(run_id):
    model_path = models_dir / f"model_{run_id}.h5"
    return model_path

def create_tensorboard_callback(run_id):
    return TensorBoard(
        log_dir=str(logs_dir / run_id),
        histogram_freq=config.histogram_freq,
        profile_batch=config.profile_batch)


def create_csv_logger_callback(run_id):
    return CSVLogger(
        filename=str(logs_dir / run_id / f'history.csv'),
        append=True,
        separator=';')


def create_hp_params_callback(run_id, hparams):
    return hp.KerasCallback(str(logs_dir / run_id), hparams)


def create_early_stopping_callback():
    return EarlyStopping(monitor='val_loss',
                         min_delta=0,
                         patience=3,
                         verbose=1,
                         restore_best_weights=True),


def create_model_checkpoint_callback(run_id):
    return ModelCheckpoint(str(models_dir / f'm{run_id}_checkpoint.h5'),
                           monitor='val_loss',
                           mode='min',
                           save_best_only=True,
                           verbose=1)