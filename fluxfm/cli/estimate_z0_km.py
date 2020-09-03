#!/usr/bin/env python

import sys
import os
import argparse
import configparser
import textwrap
import warnings

import numpy as np
import pandas as pd

from fluxfm.ffm_kormann_meixner import estimateZ0

def getCmdArgs():
    p = argparse.ArgumentParser(description='Simple script to read a CSV file of required variables and estimate roughness length using equations from Kormann & Meixner 2001.')

    p.add_argument('-E', '--example_ini', dest='example_ini', action='store_true', help='Write an example INI file using the name given by PROCESS_CONTROL_FILE.')
    p.add_argument(dest='in_pcf', metavar='PROCESS_CONTROL_FILE', help='Process-Control File (PCF) in INI format that provides parameter values for estimating roughness length z0 and writing output CSV file.')

    cmdargs = p.parse_args()
    return cmdargs

def getSampleIni():
    """Write a sample INI file
    
    Parameters
    ----------

    Returns
    -------
    
    """
    ini = """
    [meta_variables]
    
    [input_files]
    ; Input CSV file with the first row being header, each column being an
    ; input variable defined in the section of [input_variables].
    input_csv = /this/is/a/path/to/my/input/csv/file

    [input_variables]
    ; Column name of each required variable to estimate z0.
    ; Column name of time stamp of one observation in a row
    col_timestamp = Timestamp
    
    ; Column name of height of receptor/measurement, meter
    col_receptor_height = zm
    
    ; Column name of mean wind speed, m*s^-1
    col_alongwind_speed = wind_speed
    
    ; Column name of friction velocity, m*s^-1
    col_friction_velocity = u_
    
    ; Column name of Monin-Obukhov length, meter
    col_obukhov_length = L
    
    ; Column name of wind direction, degree
    col_wind_direction = wind_dir

    ; Column name of QC, unitless
    col_qc = qc_Tau

    ; QC maximum, only process QC <= the given maximum
    qc_max = 1
    
    ; Half window size of wind direction angles, degree. If >= 1 degree, we
    ; write median of z0 estimates from observations within wind directions of
    ; +/- "wind_direction_half_window" for each degree.
    wind_direction_half_window = 21

    ; Half temporal window size, days. If >= 1 day, for median filter, we group
    ; observations within a period of +/- temporal_half_window for each target
    ; date.
    temporal_half_window = 0

    ; Target date or dates to process and output. You are responsible to make
    ; sure rows in the input CSV cover entire temporal window centered at all 
    ; the target dates.
    target_dates = 2018-01-01
                  2018-01-02

    [output_files]
    ; Output CSV of estimated z0
    output_csv = /this/is/the/path/to/my/output/csv/file 

    [user_runtime_parameters]

    """
    return textwrap.dedent(ini)

def main(cmdargs):
    in_pcf = cmdargs.in_pcf
    out_example = cmdargs.example_ini
    if out_example:
        if os.path.isfile(in_pcf):
            msg_str = '''
            {0:s} exists! To use -E/--example_ini option to write an example
            INI to this file, delete it first! if you still need this file, did
            you type the option or file name wrong? 
            '''.format(in_pcf)
            msg_str = textwrap.dedent(msg_str)
            raise RuntimeError(msg_str)
        else:
            with open(in_pcf, 'w') as fobj:
                fobj.write(getSampleIni())
            return 0

    pcf = configparser.ConfigParser(allow_no_value=True)
    pcf.read(in_pcf)
    
    in_csv = pcf.get('input_files', 'input_csv')
    cols = dict()
    cols['ts'] = pcf.get('input_variables', 'col_timestamp')
    cols['zm'] = pcf.get('input_variables', 'col_receptor_height')
    cols['ws'] = pcf.get('input_variables', 'col_alongwind_speed')
    cols['ustar'] = pcf.get('input_variables', 'col_friction_velocity')
    cols['mo_len'] = pcf.get('input_variables', 'col_obukhov_length')
    cols['wd'] = pcf.get('input_variables', 'col_wind_direction')
    cols['qc'] = pcf.get('input_variables', 'col_qc')
    qc_max = pcf.getint('input_variables', 'qc_max')
    half_wd_win = pcf.getfloat('input_variables', 'wind_direction_half_window')
    half_time_win = pcf.getfloat('input_variables', 'temporal_half_window')
    target_dates = pcf.get('input_variables', 'target_dates').split('\n')
    out_csv = pcf.get('output_files', 'output_csv')

    df = pd.read_csv(in_csv, index_col=cols['ts'], usecols=cols.values())
    df.index = pd.DatetimeIndex(df.index)

    for td in target_dates:
        if (len(df.loc[td:td, :])==0):
            warnings.warn('Target date {0:s} is not within the range of dates in the input CSV file'.format(target_date))

    zm = df[cols['zm']].values
    ws = df[cols['ws']].values
    wd = df[cols['wd']].values
    ustar = df[cols['ustar']].values
    mo_len = df[cols['mo_len']].values
    qc = df[cols['qc']].values

    sflag = qc <= qc_max
    for var in [zm, ws, wd, ustar, mo_len]:
        sflag = np.logical_and(sflag, np.logical_not(np.isnan(var)))

    z0 = np.zeros_like(zm) + np.nan
    z0[sflag] = estimateZ0(zm[sflag], ws[sflag], wd[sflag], ustar[sflag], mo_len[sflag], wd_win=2*half_wd_win+1)

    df['z0'] = z0
    with open(out_csv, 'w') as fobj:
        for td in target_dates:
            df.loc[td:td, 'z0'].to_csv(fobj, mode='a', na_rep="NaN", date_format="%Y-%m-%d %H:%M")

if __name__ == "__main__":
    cmdargs = getCmdArgs()
    main(cmdargs)
