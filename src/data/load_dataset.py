from pathlib import Path
from typing import Any, Union, Tuple

from absl import logging
import tensorflow as tf
import os
from src import utils
from src.config import Config
from src.utils import LabelMapper


class DataInfo:
    def __init__(self, dir_path: Path, batch_size, cache_file_path: Path):
        self.dir_path = dir_path
        self.count = utils.images_count(dir_path)
        self.dirs_list = utils.get_dirs(dir_path)
        self.cache_file_path = cache_file_path
        if batch_size > self.count:
            self.batch_size = 1
        else:
            self.batch_size = batch_size
        logging.info(f"DataInfo => dir_path: {self.dir_path}, count: {self.count}, cache_file_path: {cache_file_path}")


def get_label_mapper(data_infos: [DataInfo]):
    all_classes = []
    for data_info in data_infos:
        all_classes.extend(data_info.dirs_list)
    # Return unique classes
    classes = list(set(all_classes))
    return LabelMapper(classes)


class ImageDataLoader:

    def __init__(self, config: Config):
        self.logger = config.logger
        self.number_of_channels = utils.get_number_of_channels(config.greyscale)
        self.train_data_info = DataInfo(config.train_data_path, config.batch_size, config.proccesed_train_dataset_file_path)
        self.test_data_info = DataInfo(config.test_data_path, config.batch_size, config.proccessed_test_dataset_file_path)
        self.input_shape = utils.get_input_shape(config.image_shape, self.number_of_channels)
        self.image_shape = config.image_shape
        self.labbel_mapper = get_label_mapper([self.train_data_info, self.test_data_info])
        self.train_data = self.load_dataset(data_info=self.train_data_info)
        self.test_data = self.load_dataset(data_info=self.test_data_info)

    def load_dataset(self, data_info: DataInfo):
        list_ds = tf.data.Dataset.list_files(str(data_info.dir_path / '*/*.jpg'))
        labeled_ds = list_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = self.prepare_for_training(labeled_ds, data_info)
        self.examine(ds)
        return ds

    def examine(self, labeled_ds):
        first_batch = labeled_ds.take(1)
        for images, labels in first_batch:
            tf.print(f"BatchInfo => Count of images in one batch:{images.numpy().shape[0]}", output_stream=self.logger.log_file_path)
            for image, label in zip(images, labels):
                tf.print("One image Shape: ", image.numpy().shape,
                         output_stream=self.logger.log_file_path)
                tf.print("Label: ", label.numpy(), "->", self.labbel_mapper.classes_name[label.numpy()],
                         output_stream=self.logger.log_file_path)

    def process_path(self, file_path):
        label = self.parse_label(file_path)
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
        #tf.print("file_path: ",file_path, output_stream=self.logger.log_file_path)
        label_name = tf.strings.split(file_path, os.path.sep)[-2]
        #tf.print("label_name: ", label_name, output_stream=self.logger.log_file_path)
        label = (label_name == self.labbel_mapper.classes_name)
        #tf.print("one_hot_label: ", label, output_stream=self.logger.log_file_path)
        return label

    def prepare_for_training(self, ds, data_info: DataInfo):
        # IF: small dataset, only load it once, and keep it in memory.
        # ELSE: use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        # filename: A tf.string scalar tf.Tensor, representing the name of a directory on the filesystem to use for caching elements in this Dataset.
        cache_file_path = data_info.cache_file_path
        if cache_file_path:
            ds = ds.cache(str(cache_file_path))
        else:
            ds = ds.cache()

        ds = ds.shuffle(buffer_size=data_info.count)

        # Repeat forever
        ds = ds.repeat()
        logging.info(f"batch_size:{data_info.batch_size}")
        ds = ds.batch(data_info.batch_size)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds
