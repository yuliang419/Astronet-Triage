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

import tensorflow as tf
from astronet import models
from astronet.util import config_util
from astronet.util import configdict
from astronet.data.preprocess import read_and_process_light_curve
from sklearn.metrics import roc_curve

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", type=str, default='AstroCNNModel', help="Name of the model class.")

parser.add_argument(
    "--config_name",
    type=str,
    default='local_global',
    help="Name of the model and training configuration. Exactly one of "
    "--config_name or --config_json is required.")

parser.add_argument(
    "--config_json",
    type=str,
    help="JSON string or JSON file containing the model and training "
    "configuration. Exactly one of --config_name or --config_json is required.")

parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Directory for model checkpoints and summaries.")

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
    "--suffix",
    type=str,
    default='',
    help="Suffix to add to output file names")


def find_tce(kepid, sector):
    for filename in filenames:
        for record in tf.python_io.tf_record_iterator(filename):
          ex = tf.train.Example.FromString(record)
          if (ex.features.feature["tic_id"].int64_list.value[0] == kepid) \
          and (ex.features.feature["Sectors"].int64_list.value[0] == sector):
            # print("Found {}_{} in file {}".format(kepid, tce_plnt_num, filename))
            return ex
    raise ValueError("{} not found in files: {}".format(kepid, filenames))


def plot_tce(kepid, sector, true, pred, save_dir='astronet/plots/'):
    ex = find_tce(kepid, sector)
    global_view = np.array(ex.features.feature["global_view"].float_list.value)
    local_view = np.array(ex.features.feature["local_view"].float_list.value)
    # global_centroid = np.array(ex.features.feature["global_centroid"].float_list.value)
    # local_centroid = np.array(ex.features.feature["local_centroid"].float_list.value)
    fig, axes = plt.subplots(1,2, figsize=(15, 5))
    axes[0].plot(global_view, ".")
    axes[1].plot(local_view, ".", label=true+' - '+str(pred) + ', ' + str(sector))
    plt.legend()
    plt.savefig(save_dir + str(kepid) + '.png', bbox_inches='tight')
    plt.close('all')


def plot_raw_lc(kepid, start_time, end_time, kepler_dir='astronet/kepler/'):
    time, flux = read_and_process_light_curve(kepid, kepler_dir, start_time, end_time, max_gap_width=0.75)
    fig = plt.figure(figsize=(15,4))
    plt.plot(time, flux, '.')
    plt.show()


def main(_):
    # Look up the model class.
    model_class = models.get_model_class(FLAGS.model)

    # Look up the model configuration.
    if (FLAGS.config_name is None) == (FLAGS.config_json is None):
      raise ValueError("Exactly one of config_name or config_json is required.")
    config = (
        models.get_model_config(FLAGS.model, FLAGS.config_name)
        if FLAGS.config_name else config_util.parse_json(FLAGS.config_json))
    config = configdict.ConfigDict(config)

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.model_dir)
    if not checkpoint_file:
      raise ValueError("No checkpoint file found in: %s" % FLAGS.model_dir)

    # Build the model.
    g = tf.Graph()
    with g.as_default():
      example_placeholder = tf.placeholder(tf.string, shape=[])
      parsed_features = tf.parse_single_example(
          example_placeholder,
          features={
              feature_name: tf.FixedLenFeature([feature.length], tf.float32)
              for feature_name, feature in config.inputs.features.items()
          })
      features = {}
      for feature_name, value in parsed_features.items():
        value = tf.expand_dims(value, 0)  # Add batch dimension.
        if config.inputs.features[feature_name].is_time_series:
          features.setdefault("time_series_features", {})[feature_name] = value
        else:
          features.setdefault("aux_features", {})[feature_name] = value

      model = model_class(
          features=features,
          labels=None,
          hparams=config.hparams,
          mode=tf.estimator.ModeKeys.PREDICT)
      model.build()
      saver = tf.train.Saver()

    with tf.Session(graph=g) as sess:
      saver.restore(sess, checkpoint_file)
      tf.logging.info("Successfully loaded checkpoint %s at global step %d.",
                      checkpoint_file, sess.run(model.global_step))


      pc_count = 0
      for filename in filenames:
          print(filename)
          y_true = []
          y_pred = []
          disp = []
          for serialized_example in tf.python_io.tf_record_iterator(filename):
              prediction = sess.run(
                  model.predictions,
                  feed_dict={example_placeholder: serialized_example})[0][0]
              ex = tf.train.Example.FromString(serialized_example)
              y_pred.append(prediction)
              if ex.features.feature['Disposition'].bytes_list.value[0] == 'PC':
                pc_count += 1

              y_true.append(ex.features.feature['Disposition'].bytes_list.value[0] in ['PC', 'EB'])
              disp.append(ex.features.feature['Disposition'].bytes_list.value[0])
              if (prediction >= 0.5) != (ex.features.feature['Disposition'].bytes_list.value[0] in ['PC','EB']):
                  print("prediction for %s = %s, true label = %s, sector=%s" % (ex.features.feature[
                                                                            "tic_id"].int64_list.value[0],
                                                                        prediction,
                                                    ex.features.feature['Disposition'].bytes_list.value[0], 
                                                    ex.features.feature['Sectors'].int64_list.value[0]))

                  if FLAGS.plot:
                      plot_tce(ex.features.feature["tic_id"].int64_list.value[0], ex.features.feature['Sectors'].int64_list.value[0], 
                        ex.features.feature['Disposition'].bytes_list.value[0], prediction)

          threshold = [0.3, 0.4, 0.5, 0.6]
          y_true = np.array(y_true)
          y_pred = np.array(y_pred)
          disp = np.array(disp)
          for t in threshold:
              tp = len(np.where((y_true == 1) & (y_pred >= t))[0])
              fp = len(np.where((y_true == 0) & (y_pred >= t))[0])
              fn = len(np.where((y_true == 1) & (y_pred < t))[0])
              precision = float(tp) / (tp + fp)
              recall = float(tp) / (tp + fn)

              num_pc = len(np.where((disp == 'PC') & (y_pred < t))[0])

              print("Threshold %s: precision=%s, recall=%s. Number of PCs in FNs = %s" % (t, precision, recall, num_pc))

          num = filename.split('-')[1]
          np.savetxt('true_vs_pred_'+FLAGS.suffix+'_'+num+'.txt', np.transpose([y_true, y_pred]), fmt='%f')
    print('Total %s PCs' %pc_count)      


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  test_name = 'test*'
  filenames = tf.gfile.Glob(os.path.join(FLAGS.tfrecord_dir, test_name))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)