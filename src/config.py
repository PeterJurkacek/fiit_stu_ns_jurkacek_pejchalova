import shutil
from pathlib import Path

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

from src import utils
from src.logger import Logger

project_path = Path('/labs').resolve()
data_raw_dir = project_path / 'data' / 'raw'
data_processed_dir = project_path / 'data' / 'processed'


def get_metrics():
    return [tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()
            ]


class Hyperparams:
    def __init__(self,
                 num_session_groups=10,
                 HP_CONV_LAYERS=hp.HParam("conv_layers", hp.IntInterval(1, 3)),
                 HP_CONV_KERNEL_SIZE=hp.HParam("conv_kernel_size", hp.Discrete([3, 5])),
                 HP_DENSE_LAYERS=hp.HParam("dense_layers", hp.IntInterval(1, 3)),
                 HP_DROPOUT=hp.HParam("dropout", hp.RealInterval(0.1, 0.4)),
                 HP_OPTIMIZER=hp.HParam("optimizer", hp.Discrete(["adam", "adagrad"]))):
        self.HP_CONV_LAYERS = HP_CONV_LAYERS
        self.HP_CONV_KERNEL_SIZE = HP_CONV_KERNEL_SIZE
        self.HP_DENSE_LAYERS = HP_DENSE_LAYERS
        self.HP_DROPOUT = HP_DROPOUT
        self.HP_OPTIMIZER = HP_OPTIMIZER
        self.num_session_groups = num_session_groups

        self.HPARAMS = [
            HP_CONV_LAYERS,
            HP_CONV_KERNEL_SIZE,
            HP_DENSE_LAYERS,
            HP_DROPOUT,
            HP_OPTIMIZER,
        ]

        self.METRICS = [
            hp.Metric(
                "epoch_accuracy",
                group="train",
                display_name="accuracy (train)",
            ),
            hp.Metric(
                "epoch_loss",
                group="train",
                display_name="loss (train)",
            ),
            hp.Metric(
                "epoch_accuracy",
                group="validation",
                display_name="accuracy (val.)",
            ),
            hp.Metric(
                "epoch_loss",
                group="validation",
                display_name="loss (val.)",
            )
        ]


class Config:
    def __init__(self,
                 experiment_name: str,
                 train_data_path: Path,
                 test_data_path: Path,
                 greyscale: bool,
                 epochs: int,
                 batch_size: int,
                 image_shape: tuple,
                 cache: bool,
                 histogram_freq: int,
                 update_freq: str,
                 profile_batch: int,
                 learning_rate: float,
                 loss: str,
                 optimizer: str,
                 padding: str,
                 hidden_activation: str,
                 output_activation: str,
                 units: int,
                 hyperparams=None):
        self.greyscale = greyscale
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.cache = cache
        self.cache_files_dir = data_processed_dir / experiment_name
        if not cache:
            shutil.rmtree(self.cache_files_dir, ignore_errors=True)
            self.cache_files_dir.mkdir(parents=True)
            proccesed_train_dataset_file_name = f"{train_data_path.parent.stem}{train_data_path.stem}.tfcache"
            self.proccesed_train_dataset_file_path = self.cache_files_dir / proccesed_train_dataset_file_name
            proccesed_test_dataset_file_name = f"{test_data_path.parent.stem}{test_data_path.stem}.tfcache"
            self.proccessed_test_dataset_file_path = self.cache_files_dir / proccesed_test_dataset_file_name
        else:
            self.proccesed_train_dataset_file_path = None
            self.proccessed_test_dataset_file_path = None

        self.epochs = epochs
        self.number_of_channels = utils.get_number_of_channels(greyscale)
        self.image_shape = image_shape
        self.input_shape = utils.get_input_shape(self.image_shape, self.number_of_channels)
        self.batch_size = batch_size
        self.hyperparams = hyperparams

        # Logger
        self.logger = Logger(logs_dir=project_path / 'logs' / experiment_name,
                             models_dir=project_path / 'models' / experiment_name,
                             cache_files_dir=self.cache_files_dir,
                             hyperparams=hyperparams,
                             histogram_freq=histogram_freq,
                             profile_batch=profile_batch,
                             update_freq=update_freq)

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.padding = padding
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.units = units


def get_experiment_0_config():
    dataset_name = 'test_dataset'
    config = Config(experiment_name="experiment_0",
                    train_data_path=data_raw_dir / dataset_name / 'TRAIN',
                    test_data_path=data_raw_dir / dataset_name / 'TEST',
                    greyscale=False,
                    epochs=3,
                    batch_size=16,
                    cache=False,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512,
                    hyperparams=Hyperparams())
    return config


def get_experiment_1_config(experiment_name='experiment_1'):
    dataset_name = 'DATASET'
    config = Config(experiment_name=experiment_name,
                    train_data_path=data_raw_dir / dataset_name / 'TRAIN',
                    test_data_path=data_raw_dir / dataset_name / 'TEST',
                    greyscale=False,
                    epochs=10,
                    batch_size=64,
                    cache=False,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512)
    return config


def get_experiment_3_config():
    train_dataset_name = 'anime'
    test_dataset_name = 'anime'
    experiment_name = "experiment_3"

    config = Config(experiment_name=experiment_name,
                    train_data_path=data_processed_dir / train_dataset_name / 'TRAIN',
                    test_data_path=data_processed_dir / test_dataset_name / 'TEST',
                    greyscale=False,
                    epochs=10,
                    batch_size=64,
                    cache=False,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512)
    return config


def get_experiment_4_config():
    dataset_name = 'DATASET'
    config = Config(experiment_name="experiment_4",
                    train_data_path=data_raw_dir / dataset_name / 'TRAIN',
                    test_data_path=data_raw_dir / dataset_name / 'TEST',
                    greyscale=False,
                    epochs=10,
                    batch_size=64,
                    cache=False,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512,
                    hyperparams=Hyperparams())
    return config


def get_experiment_5_config():
    dataset_name = 'DATASET'
    config = Config(experiment_name="experiment_5",
                    train_data_path=data_raw_dir / dataset_name / 'TRAIN',
                    test_data_path=data_raw_dir / dataset_name / 'TEST',
                    greyscale=True,
                    epochs=10,
                    batch_size=64,
                    buffer_size=1024,
                    cache=False,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512,
                    hyperparams=Hyperparams())
    return config


def get_experiment_6_config():
    dataset_name = 'DATASET'
    config = Config(experiment_name="experiment_6",
                    train_data_path=data_raw_dir / dataset_name / 'TRAIN',
                    test_data_path=data_raw_dir / dataset_name / 'TEST',
                    greyscale=True,
                    epochs=10,
                    batch_size=64,
                    cache=False,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512)
    return config

def get_experiment_7_config():
    dataset_name = 'DATASET'
    config = Config(experiment_name="experiment_7",
                    train_data_path=data_raw_dir / dataset_name / 'TRAIN',
                    test_data_path=data_raw_dir / dataset_name / 'TEST',
                    greyscale=False,
                    epochs=10,
                    batch_size=64,
                    cache=False,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512)
    return config