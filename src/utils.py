# %%
from pathlib import Path

import matplotlib.pyplot as plt
import datetime


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


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_dirs(path: Path):
    print(f"get_dirs({path})")
    return [item.stem for item in path.iterdir() if item.is_dir()]


def count_dirs(path: Path):
    return len(get_dirs(path))


def images_count(path: Path):
    return len(list(path.glob('**/*.jpg')))


def calculate_steps_per_epoch(total_num_of_samples, batch_size):
    return total_num_of_samples // batch_size
