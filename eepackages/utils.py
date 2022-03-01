from datetime import timedelta
from itertools import repeat
import math
from pathlib import Path
from time import sleep, time
from typing import Any, Callable, Dict, Optional

import ee
from pathos.multiprocessing import ProcessPool

from eepackages.multiprocessing.download_image import download_image

# /***
#  * The script computes surface water mask using Canny Edge detector and Otsu thresholding
#  * See the following paper for details: http://www.mdpi.com/2072-4292/8/5/386
#  *
#  * Author: Gennadii Donchyts (gennadiy.donchyts@gmail.com)
#  * Contributors: Nicholas Clinton (nclinton@google.com) - re-implemented otsu() using ee.Array
#  *
#  * Usage:
#  *
#  * var thresholding = require('users/gena/packages:thresholding')
#  *
#  * var th = thresholding.computeThresholdUsingOtsu(image, scale, bounds, cannyThreshold, cannySigma, minValue, ...)
#  *
#  */

#  /***
#  * Return the DN that maximizes interclass variance in B5 (in the region).
#  */


def otsu(histogram):
    histogram = ee.Dictionary(histogram)

    counts = ee.Array(histogram.get('histogram'))
    means = ee.Array(histogram.get('bucketMeans'))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = sum.divide(total)

    indices = ee.List.sequence(1, size)

    #  Compute between sum of squares, where each mean partitions the data.
    def f(i):
        aCounts = counts.slice(0, 0, i)
        aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
        aMeans = means.slice(0, 0, i)
        aMean = aMeans.multiply(aCounts).reduce(
            ee.Reducer.sum(), [0]).get([0]).divide(aCount)
        bCount = total.subtract(aCount)
        bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount)

        return aCount.multiply(aMean.subtract(mean).pow(2)).add(bCount.multiply(bMean.subtract(mean).pow(2)))

    bss = indices.map(f)

    #  Return the mean value corresponding to the maximum BSS.
    return means.sort(bss).get([-1])

# /***
#  * Compute a threshold using Otsu method (bimodal)
#  */


def computeThresholdUsingOtsu(image, scale, bounds, cannyThreshold, cannySigma, minValue, debug=False, minEdgeLength=None, minEdgeGradient=None, minEdgeValue=None):
    #  clip image edges
    mask = image.mask().gt(0).clip(bounds).focal_min(
        ee.Number(scale).multiply(3), 'circle', 'meters')

    #  detect sharp changes
    edge = ee.Algorithms.CannyEdgeDetector(image, cannyThreshold, cannySigma)
    edge = edge.multiply(mask)

    if minEdgeLength:
        connected = edge.mask(edge).lt(
            cannyThreshold).connectedPixelCount(200, True)

        edgeLong = connected.gte(minEdgeLength)

        # if debug:
        #  print('Edge length: ', ui.Chart.image.histogram(connected, bounds, scale, buckets))
        #  Map.addLayer(edge.mask(edge), {palette:['ff0000']}, 'edges (short)', false);

        edge = edgeLong

    #  buffer around NDWI edges
    edgeBuffer = edge.focal_max(ee.Number(scale), 'square', 'meters')

    if minEdgeValue:
        edgeMin = image.reduceNeighborhood(
            ee.Reducer.min(), ee.Kernel.circle(ee.Number(scale), 'meters'))

        edgeBuffer = edgeBuffer.updateMask(edgeMin.gt(minEdgeValue))

        # if debug:
        #  Map.addLayer(edge.updateMask(edgeBuffer), {palette:['ff0000']}, 'edge min', false);

    if minEdgeGradient:
        edgeGradient = image.gradient().abs().reduce(
            ee.Reducer.max()).updateMask(edgeBuffer.mask())

        edgeGradientTh = ee.Number(edgeGradient.reduceRegion(
            ee.Reducer.percentile([minEdgeGradient]), bounds, scale).values().get(0))

        # if debug:
        #  print('Edge gradient threshold: ', edgeGradientTh)

        #  Map.addLayer(edgeGradient.mask(edgeGradient), {palette:['ff0000']}, 'edge gradient', false);

        #  print('Edge gradient: ', ui.Chart.image.histogram(edgeGradient, bounds, scale, buckets))

        edgeBuffer = edgeBuffer.updateMask(edgeGradient.gt(edgeGradientTh))

    edge = edge.updateMask(edgeBuffer)
    edgeBuffer = edge.focal_max(
        ee.Number(scale).multiply(1), 'square', 'meters')
    imageEdge = image.mask(edgeBuffer)

    # if debug:
    #  Map.addLayer(imageEdge, {palette:['222200', 'ffff00']}, 'image edge buffer', false)

    #  compute threshold using Otsu thresholding
    buckets = 100
    hist = ee.Dictionary(ee.Dictionary(imageEdge.reduceRegion(
        ee.Reducer.histogram(buckets), bounds, scale)).values().get(0))

    threshold = ee.Algorithms.If(hist.contains(
        'bucketMeans'), otsu(hist), minValue)
    threshold = ee.Number(threshold)

    if debug:
        # // experimental
        # // var jrc = ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('occurrence')
        # // var jrcTh = ee.Number(ee.Dictionary(jrc.updateMask(edge).reduceRegion(ee.Reducer.mode(), bounds, scale)).values().get(0))
        # // var water = jrc.gt(jrcTh)
        # // Map.addLayer(jrc, {palette: ['000000', 'ffff00']}, 'JRC')
        # // print('JRC occurrence (edge)', ui.Chart.image.histogram(jrc.updateMask(edge), bounds, scale, buckets))

        # Map.addLayer(edge.mask(edge), {palette:['ff0000']}, 'edges', true);

        print('Threshold: ', threshold)

        # print('Image values:', ui.Chart.image.histogram(image, bounds, scale, buckets));
        # print('Image values (edge): ', ui.Chart.image.histogram(imageEdge, bounds, scale, buckets));
        # Map.addLayer(mask.mask(mask), {palette:['000000']}, 'image mask', false);

    if minValue is not None:
        return threshold.max(minValue)
    else:
        return threshold


def focalMin(image: ee.Image, radius: float):
    erosion: ee.Image = image.Not().fastDistanceTransform(
        radius).sqrt().lte(radius).Not()
    return erosion


def focalMax(image: ee.Image, radius: float):
    dilation: ee.Image = image.fastDistanceTransform(radius).sqrt().lte(radius)
    return dilation


def focalMaxWeight(image: ee.Image, radius: float):
    distance: ee.Image = image.fastDistanceTransform(radius).sqrt()
    dilation: ee.Image = distance.where(distance.gte(radius), radius)
    dilation = ee.Image(radius).subtract(dilation).divide(radius)
    return dilation


def _batch_download_ic(
    ic: ee.ImageCollection,
    img_download_method: Callable,
    name_prefix: str,
    out_dir: Optional[Path],
    pool_size: int,
    download_kwargs: Optional[Dict[str, Any]]
):
    """
    does the actual work batch downloading images in an ee.ImageCollection using the python
    multiprocessing module. Takes the img download method as a callable to implement different
    ee.Image methods.
    """

    out_dir.mkdir(exist_ok=True, parents=True)
    log_dir: Path = out_dir / "logging"
    log_dir.mkdir(exist_ok=True, parents=True)

    num_images: int = ic.size().getInfo()
    image_list: ee.List = ic.toList(num_images)

    serialized_il: str = image_list.serialize()

    pool: ProcessPool = ProcessPool(nodes=pool_size)
    result = pool.amap(download_image,
        repeat(ee),
        range(num_images),
        repeat(img_download_method),
        repeat(serialized_il),
        repeat(name_prefix),
        repeat(out_dir),
        repeat(download_kwargs)
    )

    start_time: time = time()
    elapsed: timedelta = timedelta(0)
    result.get()  # needed to trigger execution?

    while not result.ready() and elapsed.total_seconds() < 3600:  # 1h timeout
        sleep(5)
        elapsed = timedelta(seconds=time() - start_time)

    pool.close()
    pool.join()
    pool.clear()

def download_image_collection(
    ic: ee.ImageCollection,
    name_prefix: str = "ic_",
    out_dir: Optional[Path] = None,
    pool_size: int = 25,
    download_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    """
    Download images in image collection. Only works for images in the collection that are < 32M
    and grid dimension < 10000, documented at 
    https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl.

    args:
        ic (ee.ImageCollection): ImageCollection to download.
        name_prefix (str): prefix for the filename of the downloaded objects.
        out_dir (Optional(Path)): pathlib object referring to output dir.
        pool_size (int): multiprocessing pool size.
        download_kwargs (Optional(Dict(str, Any))): keyword arguments used in
            [getDownloadUrl](https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl).
    """
    _batch_download_ic(ic, ee.Image.getDownloadURL,
                       name_prefix, out_dir, pool_size, download_kwargs)


def download_image_collection_thumb(
    ic: ee.ImageCollection,
    name_prefix: str = "ic_",
    out_dir: Optional[Path] = None,
    pool_size: int = 25,
    download_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    """
    Download thumb images in and image collection. Only works for images in the collection that are < 32M
    and grid dimension < 10000, documented at
    https://developers.google.com/earth-engine/apidocs/ee-image-getthumburl.

    args:
        ic (ee.ImageCollection): ImageCollection to download.
        name_prefix (str): prefix for the filename of the downloaded objects.
        out_dir (Optional(Path)): pathlib object referring to output dir.
        pool_size (int): multiprocessing pool size.
        download_kwargs (Optional(Dict(str, Any))): keyword arguments used in
            [getDownloadUrl](https://developers.google.com/earth-engine/apidocs/ee-image-getthumburl).
    """
    _batch_download_ic(ic, ee.Image.getThumbURL, name_prefix,
                       out_dir, pool_size, download_kwargs)


def radians(img):
    """Converts image from degrees to radians"""
    return img.toFloat().multiply(math.pi).divide(180)


def hillshade(az, ze, slope, aspect):
    """Computes hillshade"""
    azimuth = radians(ee.Image.constant(az))
    zenith = radians(ee.Image.constant(90).subtract(ee.Image.constant(ze)))

    return azimuth \
        .subtract(aspect).cos().multiply(slope.sin()).multiply(zenith.sin()) \
        .add(zenith.cos().multiply(slope.cos()))


def hillshadeRGB(image, elevation, weight=1, height_multiplier=5, azimuth=0, zenith=45,
                 contrast=0, brightness=0, saturation=1, castShadows=False, customTerrain=False):
    """Styles RGB image using hillshading, mixes RGB and hillshade using HSV<->RGB transform"""

    hsv = image.visualize().unitScale(0, 255).rgbToHsv()

    z = elevation.multiply(ee.Image.constant(height_multiplier))

    terrain = ee.Algorithms.Terrain(z)
    slope = radians(terrain.select(['slope'])).resample('bicubic')
    aspect = radians(terrain.select(['aspect'])).resample('bicubic')

    if customTerrain:
        raise NotImplementedError(
            'customTerrain argument is not implemented yet')

    hs = hillshade(azimuth, zenith, slope, aspect).resample('bicubic')

    if castShadows:
        hysteresis = True
        neighborhoodSize = 256

        hillShadow = ee.Algorithms.HillShadow(elevation, azimuth,
                                              ee.Number(90).subtract(zenith), neighborhoodSize, hysteresis).float()

        hillShadow = ee.Image(1).float().subtract(hillShadow)

        # opening
        # hillShadow = hillShadow.multiply(hillShadow.focal_min(3).focal_max(6))

        # cleaning
        hillShadow = hillShadow.focal_mode(3)

        # smoothing
        hillShadow = hillShadow.convolve(ee.Kernel.gaussian(5, 3))

        # transparent
        hillShadow = hillShadow.multiply(0.7)

        hs = hs.subtract(hillShadow).rename('shadow')

    intensity = hs.multiply(ee.Image.constant(weight)) \
        .add(hsv.select('value').multiply(ee.Image.constant(1)
                                          .subtract(weight)))

    sat = hsv.select('saturation').multiply(saturation)

    hue = hsv.select('hue')

    result = ee.Image.cat(hue, sat, intensity).hsvToRgb() \
        .multiply(ee.Image.constant(1).float().add(contrast)).add(ee.Image.constant(brightness).float())

    if customTerrain:
        mask = elevation.mask().focal_min(2)

        result = result.updateMask(mask)

    return result


def getIsolines(image, levels=None):
    """Adds isolines to an ee.Image"""

    def addIso(image, level):
        crossing = image.subtract(level).focal_median(3).zeroCrossing()
        exact = image.eq(level)
        return ee.Image(level).float().mask(crossing.Or(exact)).set({'level': level})

    if not levels:
        levels = ee.List.sequence(0, 1, 0.1)

    levels = ee.List(levels)

    isoImages = ee.ImageCollection(levels.map(
        lambda l: addIso(image, ee.Number(l))))

    return isoImages
