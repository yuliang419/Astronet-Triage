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

"""Functions for reading Kepler data.
Need to figure out how to test this..."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


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


def square_error(x, model):
    """

    :param x: 1D numpy array of data values.
    :param model: 1D numpy array of model values.
    :return: sum of square of residuals
    """
    return sum((model - x)**2)


def box_model(time, t0, duration, depth):
    """
    Generate box-shaped transit model.
    :param time: 1D array of time values.
    :param t0: time of centre of event.
    :param duration: duration of event (in days).
    :param depth: depth of event.
    :return: 1D array of model flux values corresponding to input times.
    """
    model = np.ones(len(time))
    model[abs(time - t0) < duration/2] -= depth
    return model


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

        SR = -s*abs(s) / (r*(1-r))  # negative signals should give large positive s
        if SR > best_SR:
            best_t0 = t0
            best_SR = SR
    return best_t0, new_time, new_flux+1.
