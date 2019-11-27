from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import src.data.load_dataset as loader
import src.logger as logger
import src.tunning as tunning


def main():
    logger.start()
    input_shape = loader.get_input_shape()
    classes = loader.get_unique_classes()
    #trainer.start(model=get_cnn(input_shape=input_shape, classes=classes))
    tunning.start(verbose=True)
    logger.end()


if __name__ == '__main__':
    main()
