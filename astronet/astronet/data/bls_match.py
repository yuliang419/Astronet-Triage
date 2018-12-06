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

"""Code for creating catalog of "bad" training samples in the same format as TCE catalogs."""

import pandas as pd
import numpy as np
import os


base_dir = '/pdo/users/yuliang'
bad_lists = ['sector-1-bad.txt', 'sector-2-bright-bad.txt', 'sector-2-faint-bad.txt', 'sector-3-1-bad.txt']

columns = ['src', 'tic_id', 'toi_id', 'Disposition', 'RA', 'Dec', 'Tmag', 'Tmag Err', 'Epoc', 'Epoc Err', 'Period',
           'Period Err', 'Duration', 'Duration Err', 'Transit Depth', 'Transit Depth Err', 'Sectors']
bad_tces = pd.DataFrame(columns=columns)

for bad_list in bad_lists:
    new_dataframe = pd.DataFrame(columns=columns)
    tics = np.loadtxt(os.path.join(base_dir, bad_list), dtype=int)
    new_dataframe['tic_id'] = tics
    new_dataframe['src'] = 'qlp'
    new_dataframe['toi_id'] = '.01'
    new_dataframe['Sectors'] = int(bad_list.split('-')[1])
    new_dataframe['Disposition'] = 'J'
    bad_tces = pd.concat([bad_tces, new_dataframe], ignore_index=True)

bad_tces.to_csv('/pdo/users/yuliang/ebclassify/astronet/astronet/bad_tces.csv', index=False)