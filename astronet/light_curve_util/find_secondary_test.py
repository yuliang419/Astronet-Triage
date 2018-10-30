from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from light_curve_util import find_secondary as fs
from astronet.data import preprocess
import pandas as pd
import matplotlib.pyplot as plt
import sys


def match_tce(kepid, input_tce_csv_file, planet_num=1):
    tce_table = pd.read_csv(input_tce_csv_file, index_col="rowid", comment="#")
    match = tce_table[(tce_table['kepid'] == kepid) & (tce_table['tce_plnt_num'] == planet_num)]
    if len(match) == 0:
        print('TCE not found')
        return
    else:
        return match['tce_period'].values[0], match['tce_time0bk'].values[0], match['tce_duration'].values[0]/24.


def test_secondary(kepid, kepler_data_dir, input_tce_csv_file, start_time, end_time, planet_num=1):
    # type: (int, str, str, float, float, int) -> None
    period, time0bk, duration = match_tce(kepid, input_tce_csv_file, planet_num)
    time, flux = preprocess.read_and_process_light_curve(kepid, kepler_data_dir, start_time, end_time)
    time, flux = preprocess.phase_fold_and_sort_light_curve(time, flux, period, time0bk)

    t0, new_time, new_flux = fs.find_secondary(time, flux, duration, period)
    fig = plt.figure()
    plt.plot(new_time, new_flux, '.')
    plt.plot(t0, 0, 'r*')
    plt.show()


if __name__ == '__main__':
    kepid = int(sys.argv[1])
    kepler_data_dir = 'astronet/kepler'
    input_tce_csv_file = 'astronet/dr24_tce.csv'
    test_secondary(kepid, kepler_data_dir, input_tce_csv_file, 667, 697)
