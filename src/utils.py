from pathlib import Path
import logging

import matplotlib.pyplot as plt
import datetime
import time

import numpy as np
import pandas as pd

default_timeit_steps = 1000


def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy().decode('utf-8'))
    plt.axis('off')


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_loaded_data(labeled_ds, number_of_images=2):
    print(f"Loaded_number_of_images: {number_of_images}")
    for image_raw, label_text in labeled_ds.take(number_of_images):
        print(repr(image_raw.numpy()))
        print()
        print(label_text.numpy())


def timestamp():
    _timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # logging.info(f"timestamp: {timestamp}")
    return _timestamp


def get_dirs(path: Path):
    dirs = [item.stem for item in path.iterdir() if item.is_dir()]
    # print(f"DIRS: {dirs} on PATH: {path}")
    logging.info(f"dirs: {dirs}, from: {path}")
    return dirs


def count_dirs(path: Path):
    count = len(get_dirs(path))
    # logging.info(f"count_dirs: {count}")
    return count


def images_count(path: Path):
    count = len(list(jpg_images_from(path)))
    # logging.info(f"images_count: {count}")
    return count


def jpg_images_from(path: Path):
    # logging.info(f"from path: {path}")
    return path.glob('**/*.jpg')


def calculate_steps_per_epoch(total_num_of_samples, batch_size):
    steps_per_epoch = total_num_of_samples // batch_size
    logging.info(f"steps_per_epoch: {steps_per_epoch}")
    logging.info(f"total_num_of_samples: {total_num_of_samples}")
    logging.info(f"batch_size: {batch_size}")
    if steps_per_epoch == 0:
        steps_per_epoch = total_num_of_samples
    return steps_per_epoch


def timeit(ds, batch_size, steps=default_timeit_steps):
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        batch = next(it)
        if i % 10 == 0:
            print('.', end='')
    print()
    end = time.time()

    duration = end - start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(batch_size * steps / duration))


def load_result_df(file_path: Path):
    return pd.read_csv(str(file_path),
                       header=None,
                       index=None,
                       names=['Experiment', 'Trieda', 'Počet obrázkov', 'Úspešnosť'])


def get_number_of_channels(greyscale: bool):
    logging.info(f"Greyscale: {greyscale}")
    return 1 if greyscale else 3


def get_input_shape(image_shape: tuple, number_of_channels: int):
    logging.info(f"Number of channels: {number_of_channels}")
    return image_shape + (number_of_channels,)


class Label:
    def __init__(self, name: str, number: int):
        self.name = name
        self.number = number


class LabelMapper:
    def __init__(self, classes_name: [str]):
        self.classes_name = np.array(classes_name)
        self.labels = [Label(name=name, number=index) for (index, name) in enumerate(classes_name)]
        logging.info(f"labels: {[(label.name, label.number) for label in self.labels]}")
        self.name_by_number = {}
        self.num_by_name = {}
        for label in self.labels:
            self.name_by_number.update({label.number: label.name})
            self.num_by_name.update({label.name: label.number})
        logging.debug(f"self.name_by_number: {self.name_by_number}")
        logging.debug(f"self.num_by_name: {self.num_by_name}")
