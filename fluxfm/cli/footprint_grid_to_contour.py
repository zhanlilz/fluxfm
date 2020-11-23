#!/usr/bin/env python

import os
import argparse
import warnings

import numpy as np

from osgeo import gdal, osr, ogr
from affine import Affine

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

gdal.AllRegister()

def getCmdArgs():
    p = argparse.ArgumentParser(description='''Convert footprint grid to
            contour lines saved in a vector file.''')

    p.add_argument('-b', '--band', dest='band', metavar='BAND_INDEX', 
            type=int, required=False, default=1, 
            help='''Index to the band in the raster file to use for generating
            contour lines, with first band being 1. Default: 1.''')

    defval = np.arange(10, 100, 10).tolist()
    p.add_argument('-p', '--percentage', dest='percentage', 
            metavar='PERCENTAGE_LEVEL', 
            type=float, nargs='+', required=False, 
            default=defval, 
            help='''Cumulative percentage levels where to draw contour lines.
            The integral percentage of contribution to footprint within a
            contour line will be its associated percentage. Default:
            {0:s}'''.format(str(defval)))
    p.add_argument('--format', dest='out_format', 
            metavar='OUTPUT_VECTOR_FORMAT', required=False, default='GPKG', 
            help='''Format of output vector file of contour lines. Default:
            GPKG''')
    p.add_argument(dest='in_grid', metavar='INPUT_GRID', 
            help='''Input raster file of footprint grid.''')
    p.add_argument(dest='out_contour', metavar='OUTPUT_CONTOUR', 
            help='''Output vector file of contour lines.''')

    cmdargs = p.parse_args()
    return cmdargs

def main(cmdargs):
    in_grid_raster = cmdargs.in_grid
    out_contour_vector = cmdargs.out_contour
    iband = cmdargs.band
    out_format = cmdargs.out_format
    percentages = cmdargs.percentage
    plevels = np.asarray(percentages) * 0.01
    plevels = np.sort(plevels)

    raster_ds = gdal.Open(in_grid_raster, gdal.GA_ReadOnly)
    band = raster_ds.GetRasterBand(iband)
    if band is None:
        msg = 'No band found for given band index {0:d}'.format(iband)
        raise RuntimeError(msg)
    ndv = band.GetNoDataValue()
    grid_z = band.ReadAsArray()
    if grid_z.dtype.kind in np.typecodes['Float']:
        eps = np.finfo(grid_z.dtype).resolution
        zflag = np.fabs(grid_z - ndv) > eps
    else:
        eps = 1
        zflag = grid_z != ndv
    if np.sum(zflag) < 1:
        # no valid z values ...
        msg = 'No valid values of source weight in ' \
                + 'the given footprint raster {0:s}'
        msg = msg.format(in_grid_raster)
        raise RuntimeError(msg) 

    geotransform = raster_ds.GetGeoTransform()
    fwd = Affine.from_gdal(*geotransform)
    
    nx, ny = raster_ds.RasterXSize, raster_ds.RasterYSize
    ix, iy = np.meshgrid(np.arange(0.5, nx), np.arange(0.5, ny))

    grid_x, grid_y = zip(*[fwd*(xval, yval) 
        for xval, yval in zip(ix[:], iy[:])])
    grid_x = np.reshape(grid_x, ix.shape)
    grid_y = np.reshape(grid_y, iy.shape)

    # Figure out the levels of grid_z values at each level of percentage
    # (integral of grid_z values enclosed by a contour line at a level of z,
    # that is, integral of z values > a z level, because by definition, z
    # values within a contour line of z level should be all larger than this z
    # level). 
    sorted_z = np.sort(grid_z[zflag])[::-1]
    integral = np.cumsum(sorted_z)

    # It is possible that the entire domain of the input grid does not cover
    # 100% of the footprint, we can only draw contour lines to the maximum
    # percentage the entire domain contains.
    sflag = plevels <= integral.max()
    plevels = plevels[sflag]
    # If we do not have any valid plevels left ...
    if len(plevels) < 1:
        msg = 'Given levels of percentage are either too small or too large ' \
                + 'to draw contour lines from the given raster {0:s}.'
        msg = msg.format(in_grid_raster)
        raise RuntimeError(msg)
    if not np.all(sflag):
        msg = 'The extent of input raster only covers up to ' \
                + '{0:.3f}% of footprint, not large enough ' \
                + 'to cover all the given cumulative percentages ' \
                + 'of contour lines to plot.' \
                + '\nThe highest percentage of the output contour lines is ' \
                + '{1:f}'
        msg = msg.format(integral.max()*100, np.max(plevels))
        warnings.warn(msg)
    # It is also possible that the peak pixel has a very large contribution
    # percentage that we cannot plot contour lines of small given percentages. 
    sflag = plevels >= integral.min()
    plevels = plevels[sflag]
    # If we do not have any valid plevels left ...
    if len(plevels) < 1:
        msg = 'Given levels of percentage are either too small or too large ' \
                + 'to draw contour lines from the given raster {0:s}.'
        msg = msg.format(in_grid_raster)
        raise RuntimeError(msg)
    if not np.all(sflag):
        msg = 'The peak pixel of input raster covers ' \
                + '{0:.3f}% of footprint, not small enough ' \
                + 'to resolve all the given cumulative percentages ' \
                + 'of contour lines to plot.' \
                + '\nThe lowest percentage of the output contour lines is ' \
                + '{1:f}'
        msg = msg.format(integral.min()*100, np.min(plevels))
        warnings.warn(msg)
    
    zlevels = np.interp(plevels, integral, sorted_z)

    # Using matplotlib to generate contour lines.
    zlevels = zlevels[::-1]
    plevels = plevels[::-1]
    contourset = plt.contour(grid_x, grid_y, grid_z, zlevels)

    ogr_driver = ogr.GetDriverByName(out_format)
    vector_ds = ogr_driver.CreateDataSource(out_contour_vector)
    layer_name = os.path.basename(out_contour_vector)
    layer_name = '.'.join(layer_name.split('.')[:-1])
    layer = vector_ds.CreateLayer(layer_name, raster_ds.GetSpatialRef(), 
            ogr.wkbMultiLineString)
    field_def = ogr.FieldDefn('zlevel', ogr.OFTReal)
    # field_def.SetPrecision(10)
    layer.CreateField(field_def)
    layer.CreateField(ogr.FieldDefn('plevel', ogr.OFTReal))
    for i, segs in enumerate(contourset.allsegs):
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField('zlevel', zlevels[i])
        feature.SetField('plevel', plevels[i])
        geom = ogr.Geometry(ogr.wkbMultiLineString)
        for ss in segs:
            geom_seg = ogr.Geometry(ogr.wkbLineString)
            for pts in ss:
                geom_seg.AddPoint(*pts)
            geom.AddGeometry(geom_seg)
        feature.SetGeometry(geom)
        layer.CreateFeature(feature)
        feature = None

    raster_ds = None
    vector_ds = None

if __name__ == "__main__":
    cmdargs = getCmdArgs()
    main(cmdargs)
