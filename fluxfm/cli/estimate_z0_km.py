#!/usr/bin/env python

import sys
import argparse

import numpy as np
import pandas as pd

sys.path.append("/home/zhanli/Workspace/src/fluxfm/fluxfm/kormann_meixner")
from ffm_kormann_meixner import estimateZ0

def getCmdArgs():
    p = argparse.ArgumentParser(description='Simple script to read a CSV file of required variables and estimate roughness length using equations from Kormann & Meixner 2001.')

    p.add_argument('-t', '--target_date', dest='target_date', required=True, help='Target date in the format of e.g. 2013-01-02')
    p.add_argument('-w', '--wd_win', dest='wd_win', type=float, required=False, default=44, help='Window of wind direction angles for median filter.')
    p.add_argument(dest='in_csv', help="Path to the input CSV file including columns for variables: Timestamp,wind_speed,wind_dir,u_,L,st_dev_v_,qc_Tau,Timestamp_UTC,WTD")
    p.add_argument(dest='out_csv', help="Path to the output CSV file of roughness length z0 of the target date.")

    cmdargs = p.parse_args()
    return cmdargs

def main(cmdargs):
    in_csv = cmdargs.in_csv
    target_date = cmdargs.target_date
    wd_win = cmdargs.wd_win

    out_csv = cmdargs.out_csv

    df = pd.read_csv(in_csv, index_col=0)
    df.index = pd.DatetimeIndex(df.index)

    if (len(df.loc[target_date:target_date, :])==0):
        raise RuntimeError('Target date {0:s} is not within the range of dates in the input CSV file'.format(target_date))

    zm = 2.0 - df['WTD'] + 0.64
    zm = zm.values
    ws = df['wind_speed'].values
    wd = df['wind_dir'].values
    ustar = df['u_'].values
    mo_len = df['L'].values
    qc = df['qc_Tau'].values

    sflag = qc < 2
    for var in [zm, ws, wd, ustar, mo_len]:
        sflag = np.logical_and(sflag, np.logical_not(np.isnan(var)))

    z0 = np.zeros_like(zm) + np.nan
    z0[sflag] = estimateZ0(zm[sflag], ws[sflag], wd[sflag], ustar[sflag], mo_len[sflag], wd_win=wd_win)

    df['z0'] = z0
    df.loc[target_date:target_date, 'z0'].to_csv(out_csv, na_rep="NaN", date_format="%Y-%m-%d %H:%M")

if __name__ == "__main__":
    cmdargs = getCmdArgs()
    main(cmdargs)
