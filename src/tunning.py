# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Write sample summary data for the hparams plugin.

See also `hparams_minimal_demo.py` in this directory for a demo that
runs much faster, using synthetic data instead of actually training
MNIST models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os.path
import random
import shutil

from absl import app
from absl import flags
import numpy as np
from pathlib import Path
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from src import config
from src.data.load_dataset import ImageDataLoader
from src.models.keras_models import get_cnn_with

from tensorboard.plugins.hparams import api as hp

from src.logger import Logger
from src.models.trainer import Trainer
from src.utils import calculate_steps_per_epoch

if int(tf.__version__.split(".")[0]) < 2:
    # The tag names emitted for Keras metrics changed from "acc" (in 1.x)
    # to "accuracy" (in 2.x), so this demo does not work properly in
    # TensorFlow 1.x (even with `tf.enable_eager_execution()`).
    raise ImportError("TensorFlow 2.x is required to run this demo.")

flags.DEFINE_integer(
    "num_session_groups",
    config.num_session_groups,
    "The approximate number of session groups to create.",
)
flags.DEFINE_string(
    "experiment_name",
    config.experiment_name,
    "The directory to write the summary information to.",
)
flags.DEFINE_string(
    "dataset_name",
    config.dataset_name,
    "Use Dataset",
)
flags.DEFINE_integer(
    "summary_freq",
    config.summary_freq,
    "Summaries will be written every n steps, where n is the value of "
    "this flag.",
)
flags.DEFINE_integer(
    "num_epochs",
    config.epochs,
    "Number of epochs per trial.",
)


def run(loader, logger, session_id, hparams):
    Trainer(hparams=hparams,
            loader=loader,
            logger=logger,
            session_id=session_id,
            model=get_cnn_with(hparams=hparams,
                               seed=session_id,
                               input_shape=loader.get_input_shape(),
                               classes=loader.get_unique_classes()),
            epochs=flags.FLAGS.num_epochs)


def run_all(logger: Logger, verbose=False):
    """Perform random search over the hyperparameter space.

    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """
    rng = random.Random(0)
    loader = ImageDataLoader(batch_size=config.batch_size)
    logger.setup_hparams_config()

    sessions_per_group = 2
    num_sessions = flags.FLAGS.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups
    for group_index in xrange(flags.FLAGS.num_session_groups):
        hparams = {h: h.domain.sample_uniform(rng) for h in config.HPARAMS}
        hparams_string = str(hparams)
        for repeat_index in xrange(sessions_per_group):
            session_id = str(session_index)
            session_index += 1
            if verbose:
                logging.info(f"--- Running training session {session_index}, {num_sessions}")
                logging.info(hparams_string)
                logging.info("--- repeat #: %d" % (repeat_index + 1))
            run(logger=logger,
                loader=loader,
                session_id=session_id,
                hparams=hparams)


def steps_per_epoch_train(loader):
    return calculate_steps_per_epoch(loader.train_data_count, loader.batch_size)


def steps_per_epoch_validate(loader):
    return calculate_steps_per_epoch(loader.test_data_count, loader.batch_size)


def main(unused_argv):
    np.random.seed(0)
    experiment_name = flags.FLAGS.experiment_name
    logger = Logger(log_name=experiment_name)
    logger.start()
    logging.info("Saving output to %s." % logger.logs_dir)
    run_all(logger=logger, verbose=True)
    logging.info("Done. Output saved to %s." % logger.logs_dir)
    logger.end()


if __name__ == "__main__":
    app.run(main)
