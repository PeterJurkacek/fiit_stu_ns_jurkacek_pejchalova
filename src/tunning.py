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
import logging
import random

from absl import flags
from six.moves import xrange  # pylint: disable=redefined-builtin

import src.data.load_dataset as loader
from src import config
from src.models import hyper_trainer
from src.models.keras_models import get_cnn_with


def run(session_id, hparams):
    classes = loader.get_unique_classes()
    hyper_trainer.start(model=get_cnn_with(hparams=hparams, seed=session_id,
                                           classes=classes),
                        run_id=session_id, hparams=hparams)


def start(verbose=False):
    """Perform random search over the hyperparameter space.
    np.random.seed(0)
    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """
    rng = random.Random(0)
    sessions_per_group = 2
    num_sessions = config.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups
    for group_index in xrange(config.num_session_groups):
        hparams = {h: h.domain.sample_uniform(rng) for h in config.HPARAMS}
        hparams_string = str(hparams)
        for repeat_index in xrange(sessions_per_group):
            session_id = str(session_index)
            session_index += 1
            if verbose:
                logging.info(f"--- Running training session {session_index}, {num_sessions}")
                logging.info(hparams_string)
                logging.info("--- repeat #: %d" % (repeat_index + 1))
            run(session_id=session_id, hparams=hparams)
