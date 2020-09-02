#!/usr/bin/env python
#
# Read an INI file that provides inputs to run the footprint model by Kormann &
# Meixner 2001.
#
# Zhan Li, zhanli@gfz-potsdam.de
# Created: Sat Aug 29 15:50:34 CEST 2020

import sys
import argparse
import configparser

import numpy as np
import pandas as pd

import pyproj
from osgeo import gdal, gdal_array
gdal.AllRegister()

from fluxfm.ffm_kormann_meixner import estimateFootprint

def getCmdArgs():
    p = argparse.ArgumentParser(description='Simply command-line interface to the function of estimating footprints by the model of Kormann & Meixner 2001.')

    p.add_argument(dest='in_pcf', metavar='PROCESS_CONTROL_FILE', help='Process-Control File (PCF) in INI format that provides parameter values for estimating footprints and writing output images.')

    cmdargs = p.parse_args()
    return cmdargs

def main(cmdargs):
    in_pcf = cmdargs.in_pcf
    pcf = configparser.ConfigParser(allow_no_value=True)
    pcf.read(in_pcf)

    fp_model = pcf.get('meta_variables', 'footprint_model')
    fp_label = pcf.get('meta_variables', 'footprint_label')
    zm = pcf.getfloat('input_variables', 'receptor_height')
    z0 = pcf.getfloat('input_variables', 'roughness_length')
    ws = pcf.getfloat('input_variables', 'alongwind_speed')
    ustar = pcf.getfloat('input_variables', 'friction_velocity')
    mo_len = pcf.getfloat('input_variables', 'obukhov_length')
    sigma_v = pcf.getfloat('input_variables', 'crosswind_speed_sd')

    grid_srs = pcf.get('input_variables', 'grid_spatial_reference')
    grid_domain = pcf.get('input_variables', 'grid_domain').split()
    grid_domain = [float(val) for val in grid_domain]
    grid_res = pcf.getfloat('input_variables', 'grid_resolution')
    mxy = pcf.get('input_variables', 'receptor_location').split()
    mxy = [float(val) for val in mxy]
    
    wd = pcf.getfloat('optional_variables', 'wind_direction')
    north_type = pcf.get('optional_variables', 'north_for_wind_direction')

    out_fp_file = pcf.get('output_files', 'footprint_grid_file')
    out_fp_format = pcf.get('output_files', 'footprint_grid_format')

    dt_name = pcf.get('user_runtime_parameters', 'data_type')
    scale_factor = pcf.getfloat('user_runtime_parameters', 'scale_factor')
    add_offset = pcf.getfloat('user_runtime_parameters', 'add_offset')

    crs = pyproj.CRS.from_string(grid_srs)
    if north_type == 'due': 
        transformer = pyproj.Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
        proj = pyproj.Proj(crs)
        mc = proj.get_factors(*transformer.transform(*mxy)).meridian_convergence
        # Make wind direction w.r.t. true north to one w.r.t. grid north, such
        # that we can correctly rotate grid coordinates to along- and
        # cross-wind axes.
        wd = wd - mc

    grid_x, grid_y, grid_ffm = estimateFootprint(zm, z0, ws, ustar, mo_len, \
            sigma_v, grid_domain, grid_res, mxy, wd=wd)
    geotransform = [ \
            grid_x[0, 0]-0.5*grid_res, \
            grid_res, \
            0, \
            grid_y[0, 0]+0.5*grid_res, \
            0, \
            -grid_res]

    # Write array of footprint grid to a raster file
    gdal_dt = gdal.GetDataTypeByName(dt_name)
    numpy_dt = gdal_array.GDALTypeCodeToNumericTypeCode(gdal_dt)
    grid_ffm = ((grid_ffm-add_offset)/scale_factor).astype(numpy_dt)
    driver = gdal.GetDriverByName(out_fp_format)
    out_ds = driver.Create(out_fp_file, \
            grid_ffm.shape[1], grid_ffm.shape[0], 1, \
            gdal_dt)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(grid_ffm)
    out_band.SetNoDataValue(0) # nodata value
    out_band.SetDescription(fp_label) # band name
    out_band.SetMetadataItem('scale_factor', str(scale_factor))
    out_band.SetMetadataItem('add_offset', str(add_offset))
    # Set projection information for the output dataset
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(crs.to_wkt(version='WKT1_GDAL'))
    out_ds = None
    return 

if __name__ == "__main__":
    cmdargs = getCmdArgs()
    main(cmdargs)
