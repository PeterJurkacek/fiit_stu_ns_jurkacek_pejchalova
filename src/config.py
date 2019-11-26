from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

# data
dataset_name = 'DATASET'
image_shape = (224, 224)
OUTPUT_CLASSES = 2

HP_CONV_LAYERS = hp.HParam("conv_layers", hp.IntInterval(1, 3))
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3, 5]))
HP_DENSE_LAYERS = hp.HParam("dense_layers", hp.IntInterval(1, 3))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.1, 0.4))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "adagrad"]))

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
        group="validation",
        display_name="accuracy (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
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
]

image_width = 224
image_height = 224

learning_rate = 0.001
loss = 'sparse_categorical_crossentropy'
epochs = 7
batch_size = 256
buffer_size = 1024
cache = "cache_file"
# cnn & model
input_shape = (224, 224)
output_shape = 1

padding_same = 'same'

activation_relu = 'relu'
activation_sigmoid = 'sigmoid'
activation_softmax = 'softmax'

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

# logger
histogram_freq = 1
profile_batch = 3


num_session_groups = 5
experiment_name = 'tunning_parametrov_cnn'
summary_freq = 256