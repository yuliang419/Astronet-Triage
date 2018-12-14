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

"""Functions for reading and preprocessing light curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from light_curve_util import tess_io
from light_curve_util import median_filter
from light_curve_util import util
from third_party.kepler_spline import kepler_spline

cadence = 0.02043422

class EmptyLightCurveError(Exception):
    """Indicates light curve with no points in chosen time range."""
    pass


def read_and_process_light_curve(tic, tess_data_dir, sector=1, cam=4, ccd=1, injected=False, inject_dir='/pdo/users/yuliang'):
  """Reads an already detrended light curve.

  Args:
    tic: TIC id of the target star.
    tess_data_dir: Base directory containing TESS data. See
        tess_io.tess_filenames().

  Returns:
    time: 1D NumPy array; the time values of the light curve.
    flux: 1D NumPy array; the normalized flux values of the light curve.

  Raises:
    IOError: If the light curve files for this TIC ID cannot be found.
    EmptyLightCurveError: If light curve has no points in given time range.
  """
  # Read the TESS light curve.
  file_names = tess_io.tess_filenames(tic, tess_data_dir, sector=sector, cam=cam, ccd=ccd, injected=injected, inject_dir=inject_dir)
  if not file_names:
    tf.logging.info("Failed to find light curve files in %s for TIC ID %s" % (tess_data_dir, tic))
    raise IOError

  all_time, all_mag = tess_io.read_tess_light_curve(file_names)

  if len(all_time) < 1:
      tf.logging.info("Empty light curve. Skipped TIC id %s" % (tic))
      raise EmptyLightCurveError

  all_flux = 10.**(-(all_mag - np.median(all_mag))/2.5)
  return all_time, all_flux


def spline_detrend(kepid, time, flux):
    """Detrend light curve by fitting a B-spline
    Args:
        time: list of np arrays; timestamps split on gaps
        flux: list of np arrays; flux split on gaps

    Returns:
        time: 1D NumPy array; the time values of the light curve.
        flux: 1D NumPy array; the normalized flux values of the light curve.
    """
    # Logarithmically sample candidate break point spacings between 0.5 and 20
    # days.
    bkspaces = np.logspace(np.log10(0.5), np.log10(20), num=20)
    # Generate spline.
    spline = kepler_spline.choose_kepler_spline(
        time, flux, bkspaces, penalty_coeff=1.0, verbose=False)[0]

    if spline is None:
        raise ValueError("Failed to fit spline with Kepler ID %s", kepid)

    # Concatenate the piecewise light curve and spline.
    time = np.concatenate(time)
    flux = np.concatenate(flux)
    spline = np.concatenate(spline)

    # In rare cases the piecewise spline contains NaNs in places the spline could
    # not be fit. We can't normalize those points if the spline isn't defined
    # there. Instead we just remove them.
    finite_i = np.isfinite(spline)
    if not np.all(finite_i):
        tf.logging.warn("Incomplete spline with Kepler ID %s", kepid)
        time = time[finite_i]
        flux = flux[finite_i]
        spline = spline[finite_i]

    # "Flatten" the light curve (remove low-frequency variability) by dividing by
    # the spline.
    flux /= spline

    return time, flux


# Insert GP detrending


def phase_fold_and_sort_light_curve(time, flux, period, t0):
  """Phase folds a light curve and sorts by ascending time.

  Args:
    time: 1D NumPy array of time values.
    flux: 1D NumPy array of flux values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.

  Returns:
    folded_time: 1D NumPy array of phase folded time values in
        [-period / 2, period / 2), where 0 corresponds to t0 in the original
        time array. Values are sorted in ascending order.
    folded_flux: 1D NumPy array. Values are the same as the original input
        array, but sorted by folded_time.
  """
  # Phase fold time.
  time = util.phase_fold_time(time, period, t0)

  # Sort by ascending time.
  sorted_i = np.argsort(time)
  time = time[sorted_i]
  flux = flux[sorted_i]

  return time, flux


def generate_view(time, flux, num_bins, bin_width, t_min, t_max,
                  normalize=True):
  """Generates a view of a phase-folded light curve using a median filter.

  Args:
    time: 1D array of time values, phase folded and sorted in ascending order.
    flux: 1D array of flux values.
    num_bins: The number of intervals to divide the time axis into.
    bin_width: The width of each bin on the time axis.
    t_min: The inclusive leftmost value to consider on the time axis.
    t_max: The exclusive rightmost value to consider on the time axis.
    normalize: Whether to center the median at 0 and minimum value at -1.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  view = median_filter.median_filter(time, flux, num_bins, bin_width, t_min,
                                     t_max)
  if normalize:
    view -= np.median(view)
    view /= np.abs(np.min(view))  # In pathological cases, min(view) is zero...

  return view


def global_view(time, flux, period, num_bins=201, bin_width_factor=1.2/201):
  """Generates a 'global view' of a phase folded light curve.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of period.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=period * bin_width_factor,
      t_min=-period / 2,
      t_max=period / 2)


def twice_global_view(time, flux, period, num_bins=402, bin_width_factor=1.2 / 402):
  """Generates a 'global view' of a phase folded light curve at 2x the BLS period.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  If single transit, this is pretty much identical to global_view.

  Args:
    time: 1D array of time values, sorted in ascending order, phase-folded at 2x period.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of period.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=period * bin_width_factor,
      t_min=-period,
      t_max=period)


def local_view(time,
               flux,
               period,
               duration,
               num_bins=81,
               bin_width_factor=0.16,
               num_durations=4):
  """Generates a 'local view' of a phase folded light curve.
  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    duration: The duration of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of duration.
    num_durations: The number of durations to consider on either side of 0 (the
        event is assumed to be centered at 0).
  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=duration * bin_width_factor,
      t_min=max(-period / 2, -duration * num_durations),
      t_max=min(period / 2, duration * num_durations))


def mask_transit(time, duration, period, mask_width=2, phase_limit=0.1):
    """

    :param time: 1D array of time values, folded and sorted in ascending order, with the transit located at time 0.
    :param duration: The duration of the event (in days).
    :param period: the period of the event (in days).
    :param mask_width: number of durations to mask out.
    :param phase_limit: minimum phase to search for secondary eclipse.
    :return: mask: 1D array of booleans
    """
    mask = [(abs(t) > duration*mask_width/2) and (abs(t) > period*phase_limit) for t in time]
    return np.array(mask)


def find_secondary(time, flux, duration, period, mask_width=2, phase_limit=0.1):
    """
    Mask out transits, rearrange LC such that time goes from 0 to period. Then perform grid search for most likely
    secondary eclipse. To be called after preprocess.phase_fold_and_sort_light_curve. OOT flux should be 1.
    :param time: 1D array of time values, folded and sorted in ascending order, with the transit located at time 0.
    :param flux: 1D array of fluxes.
    :param duration: The duration of the event (in days).
    :param period: the period of the event (in days).
    :param mask_width: number of durations to mask out.
    :param phase_limit: minimum phase to search for secondary eclipse.
    :return: time of centre of most likely secondary.
    """
    if period < 1:
        mask_width = 1

    mask = mask_transit(time, duration, period, mask_width, phase_limit)

    new_time = time[mask]
    new_flux = flux[mask]

    # rearrange so that time goes from 0 to period
    new_time[new_time < 0] += period
    new_index = np.argsort(new_time)
    new_time = new_time[new_index]
    new_flux = new_flux[new_index]
    new_flux -= 1.  # centre flux at zero

    # grid search for secondary. Fix duration to duration of primary.
    time_grid = np.arange(new_time[0]+duration, new_time[-1]-duration, duration*0.1)
    min_index = 0
    max_index = min_index
    best_t0 = period / 2
    best_SR = 0

    for t0 in time_grid:
        while new_time[min_index] < (t0 - duration):
            min_index += 1
        min_in_transit = min_index
        max_in_transit = min_in_transit
        while (new_time[max_index] < (t0 + duration)) and (max_index < len(new_time)):
            max_index += 1
        while new_time[min_in_transit] < (t0 - duration/2):
            min_in_transit += 1
        while new_time[max_in_transit] < (t0 + duration/2):
            max_in_transit += 1
        if max_index - min_index < 5:
            continue
        r = float(max_in_transit - min_in_transit + 1) / len(new_time)  # assuming identical uniform weights
        s = sum(new_flux[min_in_transit:max_in_transit] / float(len(new_time)))

        SR = s**2 / (r*(1-r))
        if SR > best_SR:
            best_t0 = t0
            best_SR = SR
    return best_t0, new_time, new_flux+1.


def secondary_view(time,
               flux,
               period,
               duration,
               num_bins=81,
               bin_width_factor=0.16,
               num_durations=4
               ):
    """Generates a 'local view' of a phase folded light curve, centered on phase 0.5.
      See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
      http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
      Args:
        time: 1D array of time values, sorted in ascending order, with the transit located at time 0.
        flux: 1D array of flux values.
        period: The period of the event (in days).
        duration: The duration of the event (in days).
        num_bins: The number of intervals to divide the time axis into.
        bin_width_factor: Width of the bins, as a fraction of duration.
        num_durations: The number of durations to consider on either side of 0 (the
            event is assumed to be centered at 0).
      Returns:
        1D NumPy array of size num_bins containing the median flux values of
        uniformly spaced bins on the phase-folded time axis.
      """

    t0, new_time, new_flux = find_secondary(time, flux, duration, period)
    return generate_view(
        new_time,
        new_flux,
        num_bins=num_bins,
        bin_width=bin_width_factor * duration,
        t_min=max(t0-period / 2, t0-duration * num_durations),
        t_max=min(t0+period / 2, t0+duration * num_durations)
    )
