#!/usr/bin/env python
#
# Read an INI file that provides inputs to run the footprint model by Kormann &
# Meixner 2001.
#
# Zhan Li, zhanli@gfz-potsdam.de
# Created: Sat Aug 29 15:50:34 CEST 2020

import sys
import os
import argparse
import configparser
import textwrap

import numpy as np
import pandas as pd

import pyproj
from osgeo import gdal, gdal_array
gdal.AllRegister()

from fluxfm.ffm_kormann_meixner import estimateFootprint

def getCmdArgs():
    p = argparse.ArgumentParser(description='Simply command-line interface to the function of estimating footprints by the model of Kormann & Meixner 2001.')

    p.add_argument('-E', '--example_ini', dest='example_ini', action='store_true', help='Write an example INI file using the name given by PROCESS_CONTROL_FILE.')
    p.add_argument(dest='in_pcf', metavar='PROCESS_CONTROL_FILE', help='Process-Control File (PCF) in INI format that provides parameter values for estimating footprints and writing output images.')

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
    footprint_model = kormann and meixner
    footprint_label = 202001011645 

    [input_variables]
    ; Height of receptor/measurement, meter
    receptor_height = 1.8773 

    ; Roughness length, meter
    roughness_length = 0.006 

    ; Mean wind speed, m*s^-1
    alongwind_speed = 3.4
    
    ; Friction velocity, m*s^-1
    friction_velocity = 0.23
    
    ; Monin-Obukhov length, meter
    obukhov_length = 35.79
    
    ; Standard deviation of cross-wind speed, m*s^-1
    crosswind_speed_sd = 0.618

    ; Spatial reference system (SRS) of the output grid in GDAL-supported
    ; format. See
    ; https://gdal.org/programs/gdalsrsinfo.html?highlight=srs_def#cmdoption-gdalsrsinfo-arg-srs_def
    ; for details on GDAL-supported formats of SRS. The easiest format is to use
    ; an EPSG code, for example, "epsg:32633" is the SRS of WGS 84 / UTM zone
    ; 33N. Go to https://epsg.io/ to find the EPSG code your desired SRS. Leave
    ; it empty for non-georeferenced simple coordinate system. 
    grid_spatial_reference = epsg:32633
    
    ; Domain of the output grid on which the footprint to be estimated, given in
    ; (xmin, xmax, ymin, ymax) in four values separated by at least space
    ; characters, meter in the coordinate system given by the option *grid_srs*.
    grid_domain = 360805.0
                  361645.0
                  5971365.0
                  5972205.0
    
    ; Resolution of the footprint grid for processing and output, meter.
    grid_resolution = 0.1 
    
    ; Location (x, y) of receptor/measurement in the coordinate system of the
    ; output grid in two values separated by at least space characters.
    receptor_location = 361224.023952403
                        5971784.23062857 

    [optional_variables]
    ; Mean wind direction with regard to the north designated by
    ; "north_for_wind_direction", degree.  Leave it empty for default value 0
    ; degree, that is, the given north aligns with mean wind direction.
    wind_direction = 172.29
    
    ; Type of "North" that defines the wind direction, "due" (true north) or
    ; "grid" (grid columns along north-south direction)
    north_for_wind_direction = due

    [output_files]
    ; Output GeoTiff image file of footprint
    footprint_grid_file = /this/is/the/path/to/my/raster/file/of/footprint/grid
    footprint_grid_format = GTiff

    [user_runtime_parameters]
    ; For name of data types to use, see
    ; https://gdal.org/user/raster_data_model.html#raster-band
    data_type = Float32
    scale_factor = 1
    add_offset = 0
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
