from pathlib import Path

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

path = Path('/labs').resolve()
dataset_name = 'test_dataset'
logs_name = 'tunning_parametrov_cnn_2'

# data
logs_dir = path / 'logs' / logs_name
models_dir = path / 'models' / logs_name
data_raw_dir = path / 'data' / 'raw' / dataset_name
data_raw_train_dir = data_raw_dir / 'TRAIN'
data_raw_test_dir = data_raw_dir / 'TEST'

data_processed_dir = path / 'data' / 'processed' / dataset_name
data_processed_train_dir = data_processed_dir / 'TRAIN'
data_processed_test_dir = data_processed_dir / 'TEST'

epochs = 10
batch_size = 64
buffer_size = 1024
cache = "cache_file"
num_session_groups = 10
experiment_name = 'tunning_parametrov_cnn'
summary_freq = 256

# logger
histogram_freq = 1
update_freq='epoch'
profile_batch = 3

# NN architecture
image_width = 224
image_height = 224
image_shape = (image_width, image_height)
greyscale = True
number_of_channels = (1 if greyscale else 3)
input_shape = image_shape + (number_of_channels,)
learning_rate = 0.001
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
num_units = 500
padding = 'same'
hidden_activation = 'relu'
output_activation = 'softmax'


# Hyperparams
HP_CONV_LAYERS = hp.HParam("conv_layers", hp.IntInterval(1, 3))
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3, 5]))
HP_DENSE_LAYERS = hp.HParam("dense_layers", hp.IntInterval(1, 3))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.1, 0.4))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "adagrad"]))
filters1 = 16
kernel_size1 = 3
pool_size1 = (3, 3)

filters2 = 32
kernel_size2 = 3
pool_size2 = (3, 3)

filters3 = 64
kernel_size3 = 3
pool_size3 = (3, 3)

units = 512

# resnet
resnet_input_shape = (224, 224, 3)
resnet_include_top = True
resnet_weights = 'imagenet'

HPARAMS = [
    HP_CONV_LAYERS,
    HP_CONV_KERNEL_SIZE,
    HP_DENSE_LAYERS,
    HP_DROPOUT,
    HP_OPTIMIZER,
]

METRICS = [
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
