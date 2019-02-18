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
    "--plot",
    type=bool,
    default=True,
    help="Plot light curves?")

parser.add_argument(
    "--infile",
    type=str,
    required=True,
    help="List of input TCEs")


def find_tce(tic_id, sector):
    for filename in filenames:
        for record in tf.python_io.tf_record_iterator(filename):
          ex = tf.train.Example.FromString(record)
          if (ex.features.feature["tic_id"].int64_list.value[0] == tic_id) and (ex.features.feature["Sectors"].int64_list.value[0] == sector):
            print("Found {} in file {}".format(tic_id, filename))
            return ex

    raise ValueError("{} not found in files: {}".format(tic_id, filenames))


def plot_tce(tce, save_dir='astronet/plots/'):
    tic_id = tce['tic_id']
    ex = find_tce(tic_id, tce['Sectors'])
    global_view = np.array(ex.features.feature["global_view"].float_list.value)
    local_view = np.array(ex.features.feature["local_view"].float_list.value)
    # global_centroid = np.array(ex.features.feature["global_centroid"].float_list.value)
    # local_centroid = np.array(ex.features.feature["local_centroid"].float_list.value)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].plot(global_view, ".")
    axes[1].plot(local_view, ".", label=tce['Disposition']+' '+str(tce['Sectors']))
    plt.legend(loc='upper right', fontsize=15)
    plt.savefig(save_dir + str(tic_id) + '_' + str(tce['Sectors']) + '.png', bbox_inches='tight')
    plt.close('all')


def plot_raw_lc(tic_id, start_time, end_time, tess_dir='astronet/tess/'):
    time, flux = read_and_process_light_curve(tic_id, tess_dir, start_time, end_time, max_gap_width=0.75)
    fig = plt.figure(figsize=(15,4))
    plt.plot(time, flux, '.')
    plt.show()


def main(_):
    infile = FLAGS.infile
    tce_table = pd.read_csv(infile, header=0, usecols=[0,1,2,3,7,8,9,10,11,12,13])
    tce_table = tce_table[tce_table['Disposition'] == 'PC']
    for ind, row in tce_table.iterrows():
      try:
        plot_tce(row)
      except ValueError:
        continue

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  filenames = tf.gfile.Glob(os.path.join(FLAGS.tfrecord_dir, "*-*"))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)