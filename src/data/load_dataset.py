import sys
from pathlib import Path

sys.path.append('/labs')
import tensorflow as tf
import os

from src.utils import images_count, get_dirs

#dataset_name = 'DATASET'

class ImageDataLoader:
    def __init__(self, batch_size,
                 image_shape=(224, 224),
                 dataset_name='test_dataset',
                 greyscale=False):
        # load data here
        # To load the files as a tf.data.Dataset first create a dataset of the file paths
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.train_data_path = Path(f"/labs/data/raw/{dataset_name}/TRAIN").resolve()
        self.test_data_path = Path(f"/labs/data/raw/{dataset_name}/TEST").resolve()
        self.classes = self.get_unique_classes()
        self.train_data_count = images_count(self.train_data_path)
        self.test_data_count = images_count(self.test_data_path)
        self.greyscale = greyscale

    def get_unique_classes(self):
        all_classes_from_train = get_dirs(self.train_data_path)
        all_classes_from_test = get_dirs(self.test_data_path)
        all_classes_from_train.extend(all_classes_from_test)
        # Return unique classes
        return list(set(all_classes_from_train))

    def load_train_dataset(self):
        train_ds = self.load_dataset(self.train_data_path)
        return train_ds

    def load_test_dataset(self):
        test_ds = self.load_dataset(self.test_data_path)
        return test_ds

    def load_dataset(self, dir_data_path: Path):
        list_ds = tf.data.Dataset.list_files(str(dir_data_path / '*/*.jpg'))
        #print(f"list_ds next: {self.process_path(next(iter(list_ds)))[1]}")
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        labeled_ds = list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
        ds = self.prepare_for_training(labeled_ds, shuffle_buffer_size=images_count(dir_data_path))
        return ds

    def process_path(self, file_path: str):
        # print(file_path)
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        # print(label)
        return img, label

    def decode_img(self, image):
        # 'decode_jpeg' return image as Tensor of type uint8
        image = tf.image.decode_jpeg(image, channels=self.get_number_of_channels())
        # Use 'convert_image_dtype' to convert to floats in the [0,1] range.
        image = tf.image.convert_image_dtype(image, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(image, [self.image_shape[0], self.image_shape[1]])

    def get_number_of_channels(self):
        if self.greyscale:
            # 1: output a grayscale image.
            return 1
        else:
            # 3: output an RGB image.
            return 3

    def get_input_shape(self):
        return self.image_shape + (self.get_number_of_channels(),)

    def get_label(self, file_path: str):
        # print(f"get_label: {tf.strings.split(file_path, os.path.sep)}")
        label = tf.strings.split(file_path, os.path.sep)[-2]
        if label == 'O':
            return 0
        elif label == 'R':
            return 1
        else:
            return -1

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(self.batch_size)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds
