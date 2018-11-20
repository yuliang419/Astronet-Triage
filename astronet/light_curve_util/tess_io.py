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
# import tsig
# from tsig import catalog
from qlp.util.gaia import GaiaCatalog
import pandas as pd

from tensorflow import gfile

LONG_CADENCE_TIME_DELTA_DAYS = 0.02043422  # Approximately 29.4 minutes.


def tess_filenames(tic,
                     base_dir='/pdo/qlp-data/',
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
        dir = os.path.join(base_dir, 'sector-' + str(sector), 'ffi', 'cam' + str(cam), 'ccd' + str(ccd), 'LC')
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
    flux = np.array(api[flux_key])
    quality_flag = np.where(np.array(api["QualityFlag"]) == 'G')

    # Remove outliers
    time = time[quality_flag]
    flux = flux[quality_flag]

    # Remove NaN flux values.
    valid_indices = np.where(np.isfinite(flux))
    time = time[valid_indices]
    flux = flux[valid_indices]

    if invert:
        flux *= -1

    return time, flux


def star_query(tic, catalog, gaia_catalog):
    """

    :param tic:  TIC of the target star. May be an int or a possibly zero-
          padded string.
          catalog: tsig.catalog.TIC() object
          gaia_catalog: GaiaCatalog() object

    :return: dict containing stellar parameters.
    """

    field_list = ["id", "ra", "dec", "mass", "rad", "e_rad", "teff", "e_teff", "logg", "e_logg", "tmag", "e_tmag"]
    result, _ = catalog.query_by_id(tic, ",".join(field_list))

    dtype = [(field_list[k], float) for k in xrange(len(field_list))]
    t = np.array(result, dtype=dtype)
    starparam = {}
    starparam["tmag"] = np.array(t[:]["tmag"])[0]
    starparam["e_tmag"] = np.array(t[:]["e_tmag"])[0]
    starparam["ra"] = np.array(t[:]["ra"])[0]
    starparam["dec"] = np.array(t[:]["dec"])[0]
    starparam["mass"] = np.array(t[:]["mass"])[0]
    starparam["rad"] = np.array(t[:]["rad"])[0]
    starparam["e_rad"] = np.array(t[:]["e_rad"])[0]
    starparam["teff"] = np.array(t[:]["teff"])[0]
    starparam["e_teff"] = np.array(t[:]["e_teff"])[0]
    starparam["logg"] = np.array(t[:]["logg"])[0]
    starparam["e_logg"] = np.array(t[:]["e_logg"])[0]

    result = gaia_catalog.query_by_loc(starparam["ra"], starparam["dec"], 0.02, starparam["tmag"])
    if result is not None:
        starparam["rad"] = float(result["radius_val"])
        starparam["e_rad"] = np.sqrt(
            float(result["radius_percentile_lower"]) * float(result["radius_percentile_upper"]))
        starparam["teff"] = float(result["teff_val"])
        starparam["e_teff"] = np.sqrt(
            float(result["teff_percentile_lower"]) * float(result["teff_percentile_upper"]))

    return starparam


def bls_params(tic, sector=1, cam=4, ccd=1,base_dir='/pdo/qlp-data/'):
    """

    :param tic: TIC of the target star. May be an int or a possibly zero-
          padded string.
    :param sector: Int, sector number of data.
    :param cam: Int, camera number of data.
    :param ccd: Int, CCD number of data.
    :param base_dir: Base directory containing BLS files.
    :return: dataframe containing BLS information on significant peaks.
    """
    filename = os.path.join(base_dir, 'sector-' + str(sector), 'ffi', 'cam' + str(cam), 'ccd' + str(ccd), 'BLS',
                            tic+'.blsanal')
    df = pd.read_table(filename, delimiter=' ', header=0, escapechar='#', dtype=float)
    peaks = df[(df['BLS_SignaltoPinknoise_1_0'] > 9) & (df['BLS_Qtran_1_0'] <= 0.2) & (
                df['BLS_Qingress_1_0'] < 0.5) & (df['BLS_SN_1_0'] > 7) & (df['BLS_Depth_1_0'] < 0.1) & (
                           df['BLS_fraconenight_1_0'] < 0.8)]
    return peaks

