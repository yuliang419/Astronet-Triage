# Copyright 2018 The TensorFlow Authors.
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

"""Script for examining incorrectly classified light curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from astronet import models
from astronet.util import config_util
from astronet.util import configdict
from astronet.data.preprocess import read_and_process_light_curve
from sklearn.metrics import roc_curve

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tfrecord_dir",
    type=str,
    required=True,
    help="Directory for TFRecord files")

parser.add_argument(
    "--infile",
    type=str,
    required=True,
    help="List of input TCEs")


def find_tce(tic_id, filenames):
    for filename in filenames:
        for record in tf.python_io.tf_record_iterator(filename):
          ex = tf.train.Example.FromString(record)
          if ex.features.feature["tic_id"].int64_list.value[0] == tic_id:
            print("Found {} in file {}".format(tic_id, filename))
            return ex

    raise ValueError("{} not found in files: {}".format(tic_id, filenames))


def check_tce(tce, out, filenames):
    tic_id = tce['tic_id']
    ex = find_tce(tic_id, filenames)
    global_view = np.array(ex.features.feature["global_view"].float_list.value)
    local_view = np.array(ex.features.feature["local_view"].float_list.value)

    if not all(np.isfinite(global_view)):
        print(tic_id, "NaN in global_view")
        out.append([tic_id, 'global'])
    if not all(np.isfinite(local_view)):
        print(tic_id, "NaN in local_view")
        out.append([tic_id, 'local'])
    # global_centroid = np.array(ex.features.feature["global_centroid"].float_list.value)
    # local_centroid = np.array(ex.features.feature["local_centroid"].float_list.value)
    return out


def main(_):
    infile = FLAGS.infile
    tce_table = pd.read_csv(infile, header=0, usecols=[0,1,2,3,7,8,9,10,11,12,13])
    out = []
    for ind, row in tce_table.iterrows():
      try:
        out = check_tce(row, out, filenames)
      except ValueError:
        continue
    print(out)
    return out

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  filenames = tf.gfile.Glob(os.path.join(FLAGS.tfrecord_dir, "*"))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)