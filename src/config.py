from pathlib import Path

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

from src import utils

project_path = Path('/labs').resolve()
data_raw_dir = project_path / 'data' / 'raw'
data_processed_dir = project_path / 'data' / 'processed'


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
                 buffer_size: int,
                 image_shape: tuple,
                 cache: bool,
                 summary_freq: int,
                 histogram_freq: int,
                 update_freq: str,
                 profile_batch: int,
                 learning_rate: float,
                 loss: str,
                 metrics: [],
                 optimizer: str,
                 padding: str,
                 hidden_activation: str,
                 output_activation: str,
                 units: int,
                 hyperparams=None):
        self.logs_dir = project_path / 'logs' / experiment_name
        self.models_dir = project_path / 'models' / experiment_name

        self.greyscale = greyscale
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.cache = cache

        self.epochs = epochs
        self.buffer_size = buffer_size
        self.number_of_channels = utils.get_number_of_channels(greyscale)
        self.image_shape = image_shape
        self.input_shape = utils.get_input_shape(self.image_shape, self.number_of_channels)
        self.batch_size = batch_size

        # Logger
        self.metrics = metrics
        self.optimizer = optimizer

        self.summary_freq = summary_freq
        self.histogram_freq = histogram_freq
        self.update_freq = update_freq
        self.profile_batch = profile_batch
        self.learning_rate = learning_rate
        self.loss = loss
        self.padding = padding
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.units = units
        self.hyperparams = hyperparams


def get_experiment_0_config():
    dataset_name = 'test_dataset'
    config = Config(experiment_name="experiment_0",
                    train_data_path=data_raw_dir / dataset_name / 'TRAIN',
                    test_data_path=data_raw_dir / dataset_name / 'TEST',
                    greyscale=False,
                    epochs=2,
                    batch_size=16,
                    buffer_size=1024,
                    cache=False,
                    summary_freq=500,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512,
                    hyperparams=Hyperparams())
    return config


def get_experiment_1_config():
    dataset_name = 'DATASET'
    config = Config(experiment_name="experiment_1",
                    train_data_path=data_raw_dir / dataset_name / 'TRAIN',
                    test_data_path=data_raw_dir / dataset_name / 'TEST',
                    greyscale=False,
                    epochs=7,
                    batch_size=256,
                    buffer_size=1024,
                    cache=False,
                    summary_freq=500,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512)
    return config

def get_experiment_2_config():
    dataset_name = 'DATASET'
    config = Config(experiment_name="experiment_2",
                    train_data_path=data_raw_dir / dataset_name / 'TRAIN',
                    test_data_path=data_raw_dir / dataset_name / 'TEST',
                    greyscale=False,
                    epochs=7,
                    batch_size=256,
                    buffer_size=1024,
                    cache=False,
                    summary_freq=500,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512)
    return config

def get_experiment_3_config():
    train_dataset_name = 'anime'
    test_dataset_name = 'DATASET'

    config = Config(experiment_name="experiment_3",
                    train_data_path=data_processed_dir / train_dataset_name / 'TRAIN',
                    test_data_path=data_raw_dir / test_dataset_name / 'TEST',
                    greyscale=False,
                    epochs=7,
                    batch_size=256,
                    buffer_size=1024,
                    cache=False,
                    summary_freq=500,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
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
                    epochs=2,
                    batch_size=16,
                    buffer_size=1024,
                    cache=False,
                    summary_freq=500,
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=3,
                    image_shape=(224, 224),
                    learning_rate=0.001,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    optimizer='adam',
                    padding='same',
                    hidden_activation='relu',
                    output_activation='softmax',
                    units=512,
                    hyperparams=Hyperparams())
    return config
