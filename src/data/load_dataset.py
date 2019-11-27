import logging
from pathlib import Path
import tensorflow as tf
import os
from src.utils import images_count, get_dirs
from src import config

train_data_path = config.data_raw_train_dir
test_data_path = config.data_raw_test_dir
image_shape = config.image_shape
number_of_channels = config.number_of_channels

train_data_count = images_count(train_data_path)
test_data_count = images_count(test_data_path)
batch_size = config.batch_size
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_unique_classes():
    all_classes_from_train = get_dirs(train_data_path)
    all_classes_from_test = get_dirs(test_data_path)
    all_classes_from_train.extend(all_classes_from_test)
    # Return unique classes
    return list(set(all_classes_from_train))


def load_train_dataset():
    train_ds = load_dataset(train_data_path)
    return train_ds


def load_test_dataset():
    test_ds = load_dataset(test_data_path)
    return test_ds


def load_dataset(dir_data_path: Path):
    list_ds = tf.data.Dataset.list_files(str(dir_data_path / '*/*.jpg'))
    # print(f"list_ds next: {self.process_path(next(iter(list_ds)))[1]}")
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    image_count = images_count(dir_data_path)
    logging.info(f"{dir_data_path} image_count: {dir_data_path}")
    ds = prepare_for_training(labeled_ds, shuffle_buffer_size=image_count,cache=config.cache)  # images_count(dir_data_path))
    return ds


def process_path(file_path: str):
    # print(file_path)
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    # print(label)
    return img, label


def decode_img(image):
    # 'decode_jpeg' return image as Tensor of type uint8
    image = tf.image.decode_jpeg(image, channels=number_of_channels)
    # Use 'convert_image_dtype' to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(image, [image_shape[0], image_shape[1]])


def get_input_shape():
    return config.input_shape


def get_label(file_path: str):
    label = tf.strings.split(file_path, os.path.sep)[-2]
    if label == 'O':
        return 0
    elif label == 'R':
        return 1
    else:
        return -1


def prepare_for_training(ds, cache=config.cache, shuffle_buffer_size=config.buffer_size):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    # filename: A tf.string scalar tf.Tensor, representing the name of a directory on the filesystem to use for caching elements in this Dataset.
    # If a filename is not provided, the dataset will be cached in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(config.batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    # ds = ds.prefetch(buffer_size=self.AUTOTUNE)
    return ds
