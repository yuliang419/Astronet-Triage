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

"""Functions for creating catalog of TESS TCEs, containing TIC IDs, BLS information, stellar params, sector, camera and
CCD number."""

import numpy as np
import pandas as pd
import os
import tsig
from tsig import catalog
from qlp.util.gaia import GaiaCatalog
from tsig.spacecraft import Spacecraft
from tsig.spacecraft.geometry import LevineModel
from tsig.mission import MissionProfile


def star_query(tic):
    """

    :param tic:  TIC of the target star. May be an int or a possibly zero-
          padded string.
    :param ra: RA of target star. Float.
    :param dec: Dec of target star. Float.

    :return: dict containing stellar parameters.
    """

    field_list = ["id", "mass", "ra", "dec", "rad", "e_rad", "teff", "e_teff", "logg", "e_logg", "tmag", "e_tmag"]
    result, _ = tsig_catalog.query_by_id(tic, ",".join(field_list))

    dtype = [(field_list[k], float) for k in xrange(len(field_list))]
    t = np.array(result, dtype=dtype)
    starparam = {}
    starparam["mass"] = np.array(t[:]["mass"])[0]
    starparam["rad"] = np.array(t[:]["rad"])[0]
    starparam["e_rad"] = np.array(t[:]["e_rad"])[0]
    starparam["teff"] = np.array(t[:]["teff"])[0]
    starparam["e_teff"] = np.array(t[:]["e_teff"])[0]
    starparam["logg"] = np.array(t[:]["logg"])[0]
    starparam["e_logg"] = np.array(t[:]["e_logg"])[0]
    starparam["tmag"] = np.array(t[:]["tmag"])[0]
    starparam["e_tmag"] = np.array(t[:]["e_tmag"])[0]
    starparam["ra"] = np.array(t[:]["ra"])[0]
    starparam["dec"] = np.array(t[:]["dec"])[0]

    result = gaia_catalog.query_by_loc(starparam["ra"], starparam["dec"], 0.02, starparam["tmag"])
    if result is not None:
        if not np.isnan(float(result["radius_val"])):
            starparam["rad"] = float(result["radius_val"])
            starparam["e_rad"] = np.sqrt(
                float(result["radius_percentile_lower"]) * float(result["radius_percentile_upper"]))
        if not np.isnan(float(result["teff_val"])):
            starparam["teff"] = float(result["teff_val"])
            starparam["e_teff"] = np.sqrt(
                float(result["teff_percentile_lower"]) * float(result["teff_percentile_upper"]))

    return starparam


def bls_params(tic, sector, cam, ccd, base_dir='/pdo/qlp-data/'):
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
                            str(tic)+'.blsanal')
    peaks = pd.read_table(filename, delimiter=' ', header=0, escapechar='#', dtype=float)
    # peaks = df[(df['BLS_SignaltoPinknoise_1_0'] > 9) & (df['BLS_Qtran_1_0'] <= 0.2) & (
    #             df['BLS_Qingress_1_0'] < 0.5) & (df['BLS_SN_1_0'] > 7) & (df['BLS_Depth_1_0'] < 0.1) & (
    #                        df['BLS_fraconenight_1_0'] < 0.8)]
    return peaks


def _process_tce(tce_table):
    """
    Find camera, ccd number and stellar params of target given TIC and sector, using catalogs under /scratch/tmp
    :param tce_table: Pandas dataframe containing TIC ID, RA and Dec
    :param sector: Int, sector number of data
    :return: tce with stellar params, camera and ccd columns filled
    """
    sc = Spacecraft()
    spp = 1
    model = LevineModel

    total = len(tce_table)
    tce_table['camera'] = np.nan
    tce_table['ccd'] = np.nan
    tce_table['star_rad'] = np.nan
    tce_table['star_mass'] = np.nan
    tce_table['teff'] = np.nan
    tce_table['logg'] = np.nan

    for index, tce in tce_table.iterrows():
        if index % 10 == 0:
            print 'Processed %s/%s TCEs' % (index, total)

        sc_ra, sc_dec, sc_roll = MissionProfile.pointing_to_rdr("sector%d" % tce['Sectors'], "tess_profile.cfg")
        sc.set_pointing(sc_ra, sc_dec, sc_roll)
        starparam = star_query(tce['tic_id'])
        for cam_id in xrange(1, 5):
            x, y, ccd_n = model.celestial_to_pixel(
                starparam['ra'], starparam['dec'], sc_ra, sc_dec, sc_roll,
                sc.get_cam_geometry(cam_id),
                sc.get_ccd_geometries(cam_id), spp)
            if ccd_n[0]:
                tce_table.camera.iloc[index] = cam_id
                tce_table.ccd.iloc[index] = ccd_n[0]

        tce_table.star_rad.iloc[index] = starparam['rad']
        tce_table.star_mass.iloc[index] = starparam['mass']
        tce_table.teff.iloc[index] = starparam['teff']
        tce_table.logg.iloc[index] = starparam['logg']
        if np.isnan(tce['RA']):
            tce_table.RA.iloc[index] = starparam['ra']
            tce_table.Dec.iloc[index] = starparam['dec']

        if np.isnan(tce['Period']) and tce['camera']:
            bls = bls_params(tce['tic_id'], tce['Sectors'], tce['camera'], tce['ccd'])
            tce_table.Epoc.iloc[index] = bls['BLS_Tc_1_0'].iloc[0]
            tce_table.Period.iloc[index] = bls['BLS_Period_1_0'].iloc[0]
            tce_table.Duration.iloc[index] = bls['BLS_Qtran_1_0'].iloc[0] * bls['BLS_Period_1_0'].iloc[0] * 24
            tce_table['Transit Depth'].iloc[index] = bls['BLS_Depth_1_0'].iloc[0] * 1e6
    return tce_table


if __name__ == '__main__':
    print 'Reading TIC'
    tsig_catalog = catalog.TIC()
    print 'Reading Gaia catalog'
    gaia_catalog = GaiaCatalog()

    tce_table_names = ['../sector-1-earlylook.csv', '../sector-2-bright.csv', '../sector-3-01.csv',
                       '../sector-3-02.csv']

    tce_table = pd.DataFrame()
    for table in tce_table_names:
        tces = pd.read_csv(table, header=0, usecols=[1,2,3,4,5,6,8,10,12,14,16])
        tce_table = pd.concat([tce_table, tces], ignore_index=True)

    tce_table = _process_tce(tce_table)
    tce_table.to_csv('/pdo/users/yuliang/ebclassify/astronet/astronet/tces.csv')
