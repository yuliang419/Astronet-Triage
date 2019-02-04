# AstroNet: A Neural Network for Identifying Exoplanets in Light Curves

![Transit Animation](docs/transit.gif)

## Contact

Chris Shallue: [@cshallue](https://github.com/cshallue)

## Background

This directory contains TensorFlow models and data processing code for
identifying exoplanets in astrophysical light curves. For complete background,
see [our paper](http://adsabs.harvard.edu/abs/2018AJ....155...94S) in
*The Astronomical Journal*.

For shorter summaries, see:

* ["Earth to Exoplanet"](https://www.blog.google/topics/machine-learning/hunting-planets-machine-learning/) on the Google blog
* [This blog post](https://www.cfa.harvard.edu/~avanderb/page1.html#kepler90) by Andrew Vanderburg
* [This great article](https://milesobrien.com/artificial-intelligence-gains-intuition-hunting-exoplanets/) by Fedor Kossakovski
* [NASA's press release](https://www.nasa.gov/press-release/artificial-intelligence-nasa-data-used-to-discover-eighth-planet-circling-distant-star) article

## Citation

If you find this code useful, please cite our paper:

Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep
Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet
around Kepler-90. *The Astronomical Journal*, 155(2), 94.

Full text available at [*The Astronomical Journal*](http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta).

## Code Directories

[astronet/](astronet/)

* [TensorFlow](https://www.tensorflow.org/) code for:
  * Downloading and preprocessing Kepler data.
  * Building different types of neural network classification models.
  * Training and evaluating a new model.
  * Using a trained model to generate new predictions.

[light_curve_util/](light_curve_util)

* Utilities for operating on light curves. These include:
  * Reading Kepler data from `.fits` files.
  * Applying a median filter to smooth and normalize a light curve.
  * Phase folding, splitting, removing periodic events, etc.
* In addition, some C++ implementations of light curve utilities are located in
[light_curve_util/cc/](light_curve_util/cc).

[third_party/](third_party/)

* Utilities derived from third party code.


## Walkthrough

### Install Required Packages

First, ensure that you have installed the following required packages:

* **TensorFlow** ([instructions](https://www.tensorflow.org/install/))
* **Pandas** ([instructions](http://pandas.pydata.org/pandas-docs/stable/install.html))
* **NumPy** ([instructions](https://docs.scipy.org/doc/numpy/user/install.html))
* **AstroPy** ([instructions](http://www.astropy.org/))
* **PyDl** ([instructions](https://pypi.python.org/pypi/pydl))
* **Bazel** ([instructions](https://docs.bazel.build/versions/master/install.html))
    * Optional
* **Abseil Python Common Libraries** ([instructions](https://github.com/abseil/abseil-py))
    * Optional: only required for unit tests.

### Optional: Run Unit Tests

Verify that all dependencies are satisfied by running the unit tests:

```bash
bazel test astronet/... light_curve_util/... third_party/...
```

### Download TESS Data

A *light curve* is a plot of the brightness of a star over time. We will be
focusing on light curves produced by the TESS space telescope. An example
light curve (produced by Kepler) is shown below.

![Kepler-934](docs/kepler-943.png)

To train a model to identify planets in TESS light curves, you will need a
training set of labeled *Threshold Crossing Events* (TCEs). A TCE is a periodic
signal that has been detected in a Kepler light curve, and is associated with a
*period* (the number of days between each occurrence of the detected signal),
a *duration* (the time taken by each occurrence of the signal), an *epoch* (the
time of the first observed occurrence of the signal), and possibly additional
metadata like the signal-to-noise ratio. An example TCE is shown below. The
labels are ground truth classifications (decided by humans) that indicate which
TCEs in the training set are actual planets signals and which are caused by
other phenomena.

![Kepler-934 Transits](docs/kepler-943-transits.png)

The TESS TCE lists for each sector are available on the TEV website. Download them as CSVs, and run `make_catalog.py` in the `data` folder to create a catalog that combines all sectors. e.g.:

```
python make_catalog.py --input sector-1-earlylook.csv sector-2-bright.csv sector-3-01.csv sector-3-02.csv
```
The output will be a CSV file named `tces.csv` with the following columns:

* `row_id`: Integer ID of the row in the TCE table.
* `tic_id`: TIC ID of the target star.
* `toi_id`: TCE number within the target star. These are structured funny so we'll ignore them for now.
* `Disposition`: Final disposition from group vetting.
* `RA`: RA in degrees.
* `DEC`: Dec in degrees.
* `Tmag`: TESS magnitude.
* `Epoc`: The time corresponding to the center of the first detected
event in Barycentric Julian Day (BJD) minus a constant offset.
* `Period`: Period of the detected event, in days.
* `Duration`: Duration of the detected event, in hours.
* `Transit Depth`: Transit depth in ppm.
* `Sectors`: Sector number.
* `camera`: Camera number.
* `ccd`: CCD number.
* `star_rad`, `star_mass`, `teff`, `logg`: Stellar parameters from Gaia DR2 or the TIC.

Light curves are stored as h5 files on PDO, in e.g. `/pdo/qlp-data/sector-2/ffi/cam1/ccd1/LC/ `. Download and store them in a local directory called `astronet/tess`.

If working with TCEs that are not available on TEV, start by creating a .txt file of TIC IDs of all TCEs that you wish to include, and name the file `sector-x-yyy.txt`, where `x` is the sector number and `yyy` is an optional string. Then run `make_empty_catalog.py` in the `data` directory to create a csv file with only a few columns filled in, e.g.:

```
python make_empty_catalog.py --input sector-4-bad.txt sector-4-good.txt
```

Note that the csv file created this way will have a disposition of `J` for all TCEs, because I mostly used this to create catalogs of junk that didn't make it to group vetting.

Then, run `make_catalog.py` as usual to create a CSV file with the rest of the columns filled in.


### Process TESS Data

To train a model to identify exoplanets, you will need to provide TensorFlow
with training data in
[TFRecord](https://www.tensorflow.org/guide/datasets) format. The
TFRecord format consists of a set of sharded files containing serialized
`tf.Example` [protocol buffers](https://developers.google.com/protocol-buffers/).

The command below will generate a set of sharded TFRecord files for the TCEs in
the training set. Each `tf.Example` proto will contain the following light curve
representations:

* `global_view`: Vector of length 201: a "global view" of the TCE.
* `local_view`: Vector of length 81: a "local view" of the TCE.

In addition, each `tf.Example` will contain the value of each column in the
input TCE CSV file, including transit and stellar parameters.

Disclaimer: I haven't figured out how to make Bazel work, so just use the plain python version for now.

To generate the training set:
```bash
# Use Bazel to create executable Python scripts.
#
# Alternatively, since all code is pure Python and does not need to be compiled,
# we could invoke the source scripts with the following addition to PYTHONPATH:
#     export PYTHONPATH="/path/to/source/dir/:${PYTHONPATH}"
bazel build astronet/...

# Filename containing the CSV file of TCEs in the training set.
TCE_CSV_FILE="astronet/tces.csv"

# Directory to save output TFRecord files into.
TFRECORD_DIR="astronet/tfrecord"

# Directory where light curves are located.
TESS_DATA_DIR="astronet/tess/"

# Preprocess light curves into sharded TFRecord files using 5 worker processes.
bazel-bin/astronet/data/generate_input_records \
  --input_tce_csv_file=${TCE_CSV_FILE} \
  --tess_data_dir=${TESS_DATA_DIR} \
  --output_dir=${TFRECORD_DIR} \
  --num_worker_processes=5
  
# Or run without bazel
python astronet/data/generate_input_records.py \
--input_tce_csv_file=${TCE_CSV_FILE} \
--tess_data_dir=${TESS_DATA_DIR} \
--output_dir=${TFRECORD_DIR} \
--num_worker_processes=5
```
If the optional `--make_test_set` argument is set to True, the code will generate 8 test sets instead of 8 training, 1 validation and 1 test. This is useful for creating test sets out of new data and using them to evaluate a model trained on older data.

When the script finishes you will find 8 training files, 1 validation file and
1 test file in `TFRECORD_DIR`. The files will match the patterns
`train-0000?-of-00008`, `val-00000-of-00001` and `test-00000-of-00001`
respectively.

Here's a quick description of what the script does. For a full description, see
Section 3 of [our paper](http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta).

For TESS light curves, background variability has already been removed (mostly), so the spline-fitting step in the original paper is not necessary. 

For each TCE in the input CSV table, we generate two representations of
the light curve of that star. Both representations are *phase-folded*, which
means that we combine all periods of the detected TCE into a single curve, with
the detected event centered.

Here's an example of the generated representations (Kepler-90 g) in the output.

```python
# Launch iPython (or Python) from the tensorflow_models/astronet/ directory.
ipython

In[1]:
import matplotlib.pyplot as plt
import numpy as np
import os.path
import tensorflow as tf

In[2]:
TIC_ID = 270341214
TFRECORD_DIR = "/path/to/tfrecords/dir"

In[3]:
# Helper function to find the tf.Example corresponding to a particular TCE.
def find_tce(tic_id, filenames):
  for filename in filenames:
    for record in tf.python_io.tf_record_iterator(filename):
      ex = tf.train.Example.FromString(record)
      if ex.features.feature["tic_id"].int64_list.value[0] == tic_id:
        print("Found {} in file {}".format(tic_id, filename))
        return ex
        raise ValueError("{} not found in files: {}".format(tic_id, filenames))

In[4]:
# Find sample TCE
filenames = tf.gfile.Glob(os.path.join(TFRECORD_DIR, "*"))
assert filenames, "No files found in {}".format(TFRECORD_DIR)
ex = find_tce(TIC_ID, filenames)

In[5]:
# Plot the global and local views.
global_view = np.array(ex.features.feature["global_view"].float_list.value)
local_view = np.array(ex.features.feature["local_view"].float_list.value)
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
axes[0].plot(global_view, ".")
axes[1].plot(local_view, ".")
plt.show()
```

The output should look something like this:

![Kepler 90 g Processed](docs/kep90h-localglobal.png)

### Train an AstroNet Model

The [astronet](astronet/) directory contains several types of neural
network architecture and various configuration options. This particular version is configured to detect objects that can plausibly be planets (including PCs and EBs whose stellar variability amplitudes are less than half the depths of the eclipses).
To train a convolutional
neural network to classify TESS TCEs as either "likely planet" or "not planet",
using the best configuration from
[our paper](http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta),
run the following training script:

```bash
# Directory to save model checkpoints into.
MODEL_DIR="astronet/model/"

# Run the training script.
bazel-bin/astronet/train \
  --model=AstroCNNModel \
  --config_name=local_global \
  --train_files=${TFRECORD_DIR}/train* \
  --eval_files=${TFRECORD_DIR}/val* \
  --model_dir=${MODEL_DIR}
  
# Or without Bazel
python astronet/train.py \
--model=AstroCNNModel \
--config_name=local_global \
--train_files=${TFRECORD_DIR}/train* \
--eval_files=${TFRECORD_DIR}/val* \
--model_dir=${MODEL_DIR}
  
```
Optionally, you can also run a [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
server in a separate process for real-time
monitoring of training progress and evaluation metrics.

```bash
# Launch TensorBoard server.
tensorboard --logdir ${MODEL_DIR}
```

The TensorBoard server will show a page like this:

![TensorBoard](docs/tensorboard.png)

The "loss" plot shows both training and validation losses. The optimum number of training steps is where validation loss reaches a minimum.

### Evaluate an AstroNet Model

Run the following command to evaluate a model on the test set. The result will
be printed on the screen, and a summary file will also be written to the model
directory, which will be visible in TensorBoard.

```bash
# Run the evaluation script.
bazel-bin/astronet/evaluate \
  --model=AstroCNNModel \
  --config_name=local_global \
  --eval_files=${TFRECORD_DIR}/test* \
  --model_dir=${MODEL_DIR}
  
# Or without Bazel
python astronet/evaluate.py \
--model=AstroCNNModel \
--config_name=local_global \
--eval_files=${TFRECORD_DIR}/test* \
--model_dir=${MODEL_DIR}
```

The output should look something like this:

```bash
INFO:tensorflow:Saving dict for global step 10000: accuracy/accuracy = 0.9625159, accuracy/num_correct = 1515.0, auc = 0.988882, confusion_matrix/false_negatives = 10.0, confusion_matrix/false_positives = 49.0, confusion_matrix/true_negatives = 1165.0, confusion_matrix/true_positives = 350.0, global_step = 10000, loss = 0.112445444, losses/weighted_cross_entropy = 0.11295206, num_examples = 1574.
```

To plot the misclassified TCEs and calculate precision and recall at various thresholds, do the following:

```bash
python astronet/find_incorrect.py --model_dir=${MODEL_DIR} --tfrecord_dir=${TFRECORD_DIR} --suffix=yyy
```

This produces plots of the global and local views of misclassified TCEs in a folder called `astronet/plots` and generates a text file called `true_vs_pred_yyy_0000x.txt` with two columns: the true disposition of each TCEs in the test set (0 = junk, 1 = PC or EB), the predicted probability of each TCE being a positive. To plot the precision-recall curve, run `plot_roc.py`.

### Make Predictions

Suppose you detect a weak TCE in the light curve of the Kepler-90 star, with
period 14.44912 days, duration 2.70408 hours (0.11267 days) beginning 2.2 days
after 12:00 on 1/1/2009 (the year the Kepler telescope launched). To run this
TCE though your trained model, execute the following command:

```bash
# Generate a prediction for a new TCE.
bazel-bin/astronet/predict \
  --model=AstroCNNModel \
  --config_name=local_global \
  --model_dir=${MODEL_DIR} \
  --kepler_data_dir=${TESS_DATA_DIR} \
  --kepler_id=11442793 \
  --period=14.44912 \
  --t0=2.2 \
  --duration=0.11267 \
  --output_image_file="${HOME}/astronet/kepler-90i.png"
```

The output should look like this:

```Prediction: 0.9480018```

This means the model is about 95% confident that the input TCE is a planet.
Of course, this is only a small step in the overall process of discovering and
validating an exoplanet: the model’s prediction is not proof one way or the
other. The process of validating this signal as a real exoplanet requires
significant follow-up work by an expert astronomer --- see Sections 6.3 and 6.4
of [our paper](http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta)
for the full details. In this particular case, our follow-up analysis validated
this signal as a bona fide exoplanet: it’s now called
[Kepler-90 i](https://www.nasa.gov/press-release/artificial-intelligence-nasa-data-used-to-discover-eighth-planet-circling-distant-star),
and is the record-breaking eighth planet discovered around the Kepler-90 star!

In addition to the output prediction, the script will also produce a plot of the
input representations. For Kepler-90 i, the plot should look something like
this:

![Kepler 90 h Processed](docs/kep90i-localglobal.png)
