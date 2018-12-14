# Copyright 2018 Liang Yu.
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

"""Functions for reading TESS data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import h5py
import numpy as np

from tensorflow import gfile

LONG_CADENCE_TIME_DELTA_DAYS = 0.02043422  # Approximately 29.4 minutes.


def tess_filenames(tic,
                     base_dir='/Users/liangyu/Documents/EBClassify/astronet/astronet/tess/',
                     sector=1,
                     cam=4,
                     ccd=1,
                     injected=False,
                     inject_dir='/pdo/users/yuliang',
                     check_existence=True):
    """Returns the light curve filename for a TESS target star.

    Args:
      tic: TIC of the target star. May be an int or a possibly zero-
          padded string.
      base_dir: Base directory containing Kepler data.
      sector: Int, sector number of data.
      cam: Int, camera number of data.
      ccd: Int, CCD number of data.
      injected: Bool, whether target also has a light curve with injected planets.
      injected_dir: Directory containing light curves with injected transits.
      check_existence: If True, only return filenames corresponding to files that
          exist.

    Returns:
      filename for given TIC.
    """
    tic = str(tic)

    if not injected:
        # modify this as needed
        dir = os.path.join(base_dir, 'sector-' + str(sector), 'cam' + str(cam), 'ccd' + str(ccd))
        base_name = "%s.h5" % (tic)
        filename = os.path.join(dir, base_name)
    else:
        filename = os.path.join(inject_dir, tic + '.lc')

    if not check_existence or gfile.Exists(filename):
        return filename
    return


def read_tess_light_curve(filename, flux_key='KSPMagnitude', invert=False):
    """Reads time and flux measurements for a Kepler target star.

    Args:
      filename: str name of h5 file containing light curve.
      flux_key: Key of h5 column containing detrended flux.
      invert: Whether to invert the flux measurements by multiplying by -1.

    Returns:
      time: Numpy array; the time values of the light curve.
      flux: Numpy array corresponding to the time array.
    """
    f = h5py.File(filename, "r")
    apgroup = f["LightCurve"]["AperturePhotometry"]
    bestap = apgroup.attrs["bestap"]
    api = apgroup["Aperture_%.3d" % bestap]

    time = np.array(f["LightCurve"]["BJD"])
    mag = np.array(api[flux_key])
    if 'QFLAG' in f["LightCurve"].keys():
        quality_flag = np.where(np.array(f["LightCurve"]['QFLAG']) == 0)

        # Remove outliers
        time = time[quality_flag]
        mag = mag[quality_flag]

        # Remove NaN flux values.
        valid_indices = np.where(np.isfinite(mag))
        time = time[valid_indices]
        mag = mag[valid_indices]

    else:
        valid_indices = np.where(np.isfinite(mag))
        time = time[valid_indices]
        mag = mag[valid_indices]

        # manually remove outliers
        sigma = np.std(mag)
        quality_flag = np.where(mag >= np.median(mag) - 4 * sigma)

        # Remove outliers
        time = time[quality_flag]
        mag = mag[quality_flag]


    if invert:
        mag *= -1

    return time, mag


