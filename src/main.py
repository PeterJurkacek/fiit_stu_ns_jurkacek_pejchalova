from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import src.data.load_dataset as loader
import src.logger as logger
import src.tunning as tunning

if int(tf.__version__.split(".")[0]) < 2:
    # The tag names emitted for Keras metrics changed from "acc" (in 1.x)
    # to "accuracy" (in 2.x), so this demo does not work properly in
    # TensorFlow 1.x (even with `tf.enable_eager_execution()`).
    raise ImportError("TensorFlow 2.x is required to run this demo.")


def main():
    logger.start()
    input_shape = loader.get_input_shape()
    classes = loader.get_unique_classes()
    #trainer.start(model=get_cnn(input_shape=input_shape, classes=classes))
    tunning.start(verbose=True)
    logger.end()


if __name__ == '__main__':
    main()
