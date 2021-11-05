from itertools import repeat
import multiprocessing
import os
from typing import Optional
import requests
import shutil
from pathlib import Path

import ee
from retry import retry

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
    aMean = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount);
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
    mask = image.mask().gt(0).clip(bounds).focal_min(ee.Number(scale).multiply(3), 'circle', 'meters')

    #  detect sharp changes
    edge = ee.Algorithms.CannyEdgeDetector(image, cannyThreshold, cannySigma)
    edge = edge.multiply(mask)
    
    if minEdgeLength:
        connected = edge.mask(edge).lt(cannyThreshold).connectedPixelCount(200, True)

        edgeLong = connected.gte(minEdgeLength)

        # if debug:
        #  print('Edge length: ', ui.Chart.image.histogram(connected, bounds, scale, buckets))
        #  Map.addLayer(edge.mask(edge), {palette:['ff0000']}, 'edges (short)', false);

        edge = edgeLong
    
    #  buffer around NDWI edges
    edgeBuffer = edge.focal_max(ee.Number(scale), 'square', 'meters')

    if minEdgeValue:
      edgeMin = image.reduceNeighborhood(ee.Reducer.min(), ee.Kernel.circle(ee.Number(scale), 'meters'))

      edgeBuffer = edgeBuffer.updateMask(edgeMin.gt(minEdgeValue))

      # if debug:
      #  Map.addLayer(edge.updateMask(edgeBuffer), {palette:['ff0000']}, 'edge min', false);

    if minEdgeGradient:
      edgeGradient = image.gradient().abs().reduce(ee.Reducer.max()).updateMask(edgeBuffer.mask())

      edgeGradientTh = ee.Number(edgeGradient.reduceRegion(ee.Reducer.percentile([minEdgeGradient]), bounds, scale).values().get(0))

      # if debug:
      #  print('Edge gradient threshold: ', edgeGradientTh)
        
      #  Map.addLayer(edgeGradient.mask(edgeGradient), {palette:['ff0000']}, 'edge gradient', false);

      #  print('Edge gradient: ', ui.Chart.image.histogram(edgeGradient, bounds, scale, buckets))
      
      edgeBuffer = edgeBuffer.updateMask(edgeGradient.gt(edgeGradientTh))

    edge = edge.updateMask(edgeBuffer)
    edgeBuffer = edge.focal_max(ee.Number(scale).multiply(1), 'square', 'meters')
    imageEdge = image.mask(edgeBuffer);
    
    # if debug:
    #  Map.addLayer(imageEdge, {palette:['222200', 'ffff00']}, 'image edge buffer', false)
    
    #  compute threshold using Otsu thresholding
    buckets = 100
    hist = ee.Dictionary(ee.Dictionary(imageEdge.reduceRegion(ee.Reducer.histogram(buckets), bounds, scale)).values().get(0))

    threshold = ee.Algorithms.If(hist.contains('bucketMeans'), otsu(hist), minValue)
    threshold = ee.Number(threshold)

    if debug:
        # // experimental
        # // var jrc = ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('occurrence')
        # // var jrcTh = ee.Number(ee.Dictionary(jrc.updateMask(edge).reduceRegion(ee.Reducer.mode(), bounds, scale)).values().get(0))
        # // var water = jrc.gt(jrcTh)
        # // Map.addLayer(jrc, {palette: ['000000', 'ffff00']}, 'JRC')
        # // print('JRC occurrence (edge)', ui.Chart.image.histogram(jrc.updateMask(edge), bounds, scale, buckets))

        # Map.addLayer(edge.mask(edge), {palette:['ff0000']}, 'edges', true);

        print('Threshold: ', threshold);

        # print('Image values:', ui.Chart.image.histogram(image, bounds, scale, buckets));
        # print('Image values (edge): ', ui.Chart.image.histogram(imageEdge, bounds, scale, buckets));
        # Map.addLayer(mask.mask(mask), {palette:['000000']}, 'image mask', false);

    if minValue is not None:
      return threshold.max(minValue)
    else:
      return threshold

def focalMin(image: ee.Image, radius: float):
    erosion: ee.Image = image.Not().fastDistanceTransform(radius).sqrt().lte(radius).Not()
    return erosion

def focalMax(image: ee.Image, radius: float):
    dilation: ee.Image = image.fastDistanceTransform(radius).sqrt().lte(radius)
    return dilation

def focalMaxWeight(image: ee.Image, radius: float):
    distance: ee.Image = image.fastDistanceTransform(radius).sqrt()
    dilation: ee.Image = distance.where(distance.gte(radius), radius)
    dilation = ee.Image(radius).subtract(dilation).divide(radius)
    return dilation

@retry(tries=10, delay=5, backoff=10)
def _download_image(
    index: int,
    image_list: ee.List,
    name_prefix: str = "ic_download",
    out_dir: Optional[Path] = None
) -> None:
    """
    Hidden function to be used with download_image_collection. For the multiprocessing module, this
    function must be on the main level in the file (not within function).
    """
    if not out_dir:
        out_dir = Path.cwd() / "output"
    img: ee.Image = ee.Image(image_list.get(index))
    url: str = img.getDownloadUrl({"name": f"{name_prefix}_{index}"})
    r: requests.Response = requests.get(url, stream=True)
    filename = out_dir / f"{name_prefix}_{index}.tiff.zip"
    if not out_dir.exists():
        os.mkdir(out_dir)

    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", index)


def download_image_collection(
    ic: ee.ImageCollection,
    name_prefix: str = "ic_download",
    out_dir: Optional[Path] = None, 
    pool_size: int = 25
  ) -> None:
    """
    download images in image collection. Only works for images in the collection that are < 32M
    and grid dimension < 10000, documented at 
    https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl.

    args:
        ec (ee.ImageCollection): ImageCollection to download.
        name_prefix (str): name of the zipped_objects.
        out_dir: (Optional(Path)): pathlib object referring to output dir
        pool_size: multiprocessing pool size.
    """
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    num_images: int = ic.size().getInfo()

    image_list: ee.List = ic.toList(num_images)
    
    pool: multiprocessing.Pool = multiprocessing.Pool(pool_size)
    pool.starmap(_download_image, zip(range(num_images), repeat(image_list), repeat(name_prefix), repeat(out_dir)))

    pool.close()
    
    # Reset to default API url
    ee.Initialize()