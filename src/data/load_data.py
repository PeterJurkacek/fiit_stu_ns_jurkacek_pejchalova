import sys
from pathlib import Path

sys.path.append('/labs')
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.utils import images_count, get_dirs


class DataLoader:
    def __init__(self, batch_size, image_width=224, image_height=224):
        # load data here
        # To load the files as a tf.data.Dataset first create a dataset of the file paths
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.BATCH_SIZE = batch_size
        self.IMG_WIDTH = image_width
        self.IMG_HEIGHT = image_height
        trainDataDir = Path('/labs/data/raw/test_dataset/TRAIN')
        testDataDir = Path('/labs/data/raw/test_dataset/TEST')
        self.train_data_path = trainDataDir.resolve()
        self.test_data_path = testDataDir.resolve()
        self.train_image_generator = ImageDataGenerator(rescale=1. / self.IMG_WIDTH)  # Generator for our training data
        self.validation_image_generator = ImageDataGenerator(
            rescale=1. / self.IMG_HEIGHT)  # Generator for our validation data
        # Unique classes
        self.classes = self.get_classes()
        self.train_data_count = images_count(self.train_data_path)
        self.test_data_count = images_count(self.train_data_path)

    def get_classes(self):
        a = get_dirs(self.train_data_path)
        print(a)
        b = get_dirs(self.test_data_path)
        print(b)
        a.extend(b)
        return list(set(a))

    def get_train_data_gen(self):
        return self.get_data_gen(self.train_data_path, self.train_image_generator)

    def get_test_data_gen(self):
        return self.get_data_gen(self.test_data_path, self.validation_image_generator)

    def get_data_gen(self, data_dir, image_gen: ImageDataGenerator):
        return image_gen.flow_from_directory(batch_size=self.BATCH_SIZE,
                                             directory=data_dir,
                                             shuffle=True,
                                             target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                             class_mode='binary')

    def load_train_test(self):
        train_ds = self.load_dataset(self.train_data_path)
        test_ds = self.load_dataset(self.test_data_path)
        return train_ds, test_ds

    def load_dataset(self, dir_data_path: Path):
        list_ds = tf.data.Dataset.list_files(str(dir_data_path / '*/*.jpg'))
        print(f"list_ds next: {next(iter(list_ds))}")
        print(f"list_ds next: {self.process_path(next(iter(list_ds)))[1]}")
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        labeled_ds = list_ds.map(self.process_path)
        train_ds = self.prepare_for_training(labeled_ds, shuffle_buffer_size=10)
        # self.show_loaded_data(train_ds)
        return train_ds

    def show_loaded_data(self, labeled_ds, number_of_images=2):
        print(f"PRINTING loaded_number_of_images: {number_of_images}")
        for image_raw, label_text in labeled_ds.take(number_of_images):
            print(repr(image_raw.numpy()))
            print()
            print(label_text.numpy())

    def get_label(self, file_path):
        print(f"get_label: {tf.strings.split(file_path, '/')}")
        label = tf.strings.split(file_path, '/')[-2]
        if (label == 'O'):
            return 0
        elif (label == 'R'):
            return 1
        else:
            return -1

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    def process_path(self, file_path):
        print(file_path)
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        print(label)
        return img, label

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]

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

        ds = ds.batch(self.BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds


def test():
    datagen = DataLoader()
    try:
        print('HELLO LOAD DATA')
        train_ds, test_ds = datagen.load_train_test()
        image_raw, label_text = next(iter(train_ds))
        print([arr.numpy() for arr in label_text])

        image_raw2, label_text2 = next(iter(train_ds))
        print([arr.numpy() for arr in label_text2])
        # for image_raw, label_text in datagen.load_dataset().take(1):
        #     print([arr.numpy() for arr in label_text])
        #     #print(repr(image_raw.numpy()))
        #     #print()
        #     #print(label_text.numpy())
        return 'Funguje to'
    except Exception as e:
        return 'SA to posralo' + e

# test()
