import logging
from pathlib import Path
from typing import Any, Union, Tuple

import tensorflow as tf
import os
from src import utils
from src.config import Config
from src.utils import LabelMapper


class DataInfo:
    def __init__(self, dir_path: Path):
        self.dir_path = dir_path
        self.count = utils.images_count(dir_path)
        self.dirs_list = utils.get_dirs(dir_path)


def get_label_mapper(data_infos: [DataInfo]):
    all_classes = []
    for data_info in data_infos:
        all_classes.extend(data_info.dirs_list)
    # Return unique classes
    classes = list(set(all_classes))
    return LabelMapper(classes)


class ImageDataLoader:

    def __init__(self, config: Config):
        self.number_of_channels = utils.get_number_of_channels(config.greyscale)
        self.train_data_info = DataInfo(config.train_data_path)
        self.test_data_info = DataInfo(config.test_data_path)
        self.input_shape = utils.get_input_shape(config.image_shape, self.number_of_channels)
        self.batch_size = config.batch_size
        self.image_shape = config.image_shape
        self.labbel_mapper = get_label_mapper([self.train_data_info, self.test_data_info])
        self.train_data = self.load_dataset(data_info=self.train_data_info, cache=config.cache)
        self.test_data = self.load_dataset(data_info=self.test_data_info, cache=config.cache)

    def load_dataset(self, data_info: DataInfo, cache: bool):
        list_ds = tf.data.Dataset.list_files(str(data_info.dir_path / '*/*.jpg'))
        # print(f"list_ds next: {self.process_path(next(iter(list_ds)))[1]}")
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        labeled_ds = list_ds.map(self.process_path)
        ds = self.prepare_for_training(labeled_ds, shuffle_buffer_size=data_info.count, cache=cache)
        return ds

    def process_path(self, file_path: str):
        logging.debug(f"file_path: {file_path}")
        label = self.parse_label(file_path)
        logging.debug(f"label: {label}")
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        # print(label)
        return img, label

    def decode_img(self, image):
        # 'decode_jpeg' return image as Tensor of type uint8
        image = tf.image.decode_jpeg(image, channels=self.number_of_channels)
        # Use 'convert_image_dtype' to convert to floats in the [0,1] range.
        image = tf.image.convert_image_dtype(image, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(image, [self.image_shape[0], self.image_shape[1]])

    def parse_label(self, file_path: str):
        #logging.debug(f"file_path: {file_path}")
        label_name = tf.strings.split(file_path, os.path.sep)[-2]
        #logging.debug(f"label_name: {label_name}")
        # lebel_number = self.labbel_mapper.num_by_name[label_name]
        # logging.debug(f" lebel_number: {lebel_number}")
        return label_name == self.labbel_mapper.classes_name

    def prepare_for_training(self, ds, cache: bool, shuffle_buffer_size: int):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        # filename: A tf.string scalar tf.Tensor, representing the name of a directory on the filesystem to use for caching elements in this Dataset.
        # If a filename is not provided, the dataset will be cached in memory.
        if cache:
            ds = ds.cache()
        else:
            ds = ds.cache("cache_file_name")

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(self.batch_size)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        # ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds
