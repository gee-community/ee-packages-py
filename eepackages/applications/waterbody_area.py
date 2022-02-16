import ee
from eepackages import assets
from eepackages import utils

def computeSurfaceWaterArea(waterbody, start_filter, start, stop, scale, waterOccurrence, opt_missions):
  geom = ee.Feature(waterbody).geometry()
  
  missions = ['L4', 'L5', 'L7', 'L8', 'S2']

  if opt_missions:
    missions = opt_missions
  
  images = assets.getImages(geom, {
    'resample': True,
    'filter': ee.Filter.date(start_filter, stop),
    'missions': missions,
    'scale': scale * 10
  })
  
  # print('Image count: ', images.size())
  
  options = {
      # 'cloudFrequencyThresholdDelta': -0.15
     'scale': scale * 5
  }

  g = geom.buffer(300, scale)

  # images = assets.getMostlyCleanImages(images, g, options).sort('system:time_start')
  # images = images.filterDate(start, stop)

  # all images, but with quality score
  images = (assets.addQualityScore(images, g, options)
    .filter(ee.Filter.gt('quality_score', 0))) # sometimes null?!


  # print('Image count (clean): ', images.size())

  water = images.map(lambda i: computeSurfaceWaterArea_SingleImage(i, waterbody, scale, waterOccurrence))
  
  water = water.filter(ee.Filter.neq('area', 0))

  return water

def computeSurfaceWaterArea_SingleImage(i, waterbody, scale, waterOccurrence):
  geom = ee.Feature(waterbody).geometry()
  
  fillPercentile = 50 # // we don't trust our prior

  ndwiBands = ['green', 'swir']
  # var ndwiBands = ['green', 'nir'] 

  waterMaxImage = ee.Image().float().paint(waterbody.buffer(150), 1)
  
  maxArea = waterbody.area(scale)

  t = i.get('system:time_start')
  
  i = (i
    .updateMask(waterMaxImage)
    .updateMask(i.select('swir').min(i.select('nir')).gt(0.001)))
  
  ndwi = i.normalizedDifference(ndwiBands)

  # var water = ndwi.gt(0)

  th = utils.computeThresholdUsingOtsu(ndwi, scale, geom, 0.5, 0.7, -0.2)
  water = ndwi.gt(th)
  
  area = (ee.Image.pixelArea().mask(water)
    .reduceRegion(**{
      'reducer': ee.Reducer.sum(),
      'geometry': geom,
      'scale': scale
    }).get('area'))

  # # fill missing, estimate water probability as the lowest percentile.
  # var waterEdge = ee.Algorithms.CannyEdgeDetector(water, 0.1, 0)
  waterEdge = ee.Algorithms.CannyEdgeDetector(ndwi, 0.5, 0.7)
  
  # image mask
  imageMask = ndwi.mask()
  
  # var imageMask = i.select(ndwiBands).reduce(ee.Reducer.allNonZero())
  
  # imageMask = utils.focalMin(imageMask, ee.Number(scale) * 1.5)
  imageMask = imageMask.focal_min(ee.Number(scale).multiply(1.5), 'square', 'meters')
    #.multiply(waterMaxImage)

  # TODO: exclude non-water/missing boundsry
  waterEdge = waterEdge.updateMask(imageMask)

  # # clip by clouds
  # var bad = ee.Image(1).float().subtract(i.select('weight'))
  # bad = utils.focalMax(bad, 90).not()
  # waterEdge = waterEdge.updateMask(bad)

  # get water probability around edges
  # P(D|W) = P(D|W) * P(W) / P(D) ~=  P(D|W) * P(W)
  p = waterOccurrence.mask(waterEdge).reduceRegion(**{
    'reducer': ee.Reducer.percentile([fillPercentile]),
    'geometry': geom,
    'scale': scale
  }).values().get(0)
  
  # TODO: exclude edges belonging to cloud/water or cloud/land
  
  # TODO: use multiple percentiles (confidence margin)
    
  p = ee.Algorithms.If(ee.Algorithms.IsEqual(p, None), 101, p)

  waterFill = (waterOccurrence.gt(ee.Image.constant(p))
    .updateMask(water.unmask(0, False).Not()))
    
  # exclude false-positive, where we're sure in a non-water
  nonWater = ndwi.lt(-0.15).unmask(0, False)
  waterFill = waterFill.updateMask(nonWater.Not())
  
  fill = (ee.Image.pixelArea().mask(waterFill)
    .reduceRegion(**{
      'reducer': ee.Reducer.sum(),
      'geometry': geom,
      'scale': scale
    }).get('area'))
    
  area_filled = ee.Number(area).add(fill)
  
  filled_fraction = ee.Number(fill).divide(area_filled)

  return (i
    .addBands(waterFill.rename('water_fill'))
    .addBands(waterEdge.rename('water_edge'))
    .addBands(ndwi.rename('ndwi'))
    .addBands(water.rename('water'))
    .set({ 
      'p': p, 
      'area': area, 
      'area_filled': area_filled, 
      'filled_fraction': filled_fraction, 
      'system:time_start': t,
      'ndwi_threshold': th,
      'waterOccurrenceExpected': waterOccurrence.mask(waterEdge)
    }))

def computeSurfaceWaterAreaJRC(waterbody, start, stop, scale):
  geom = ee.Feature(waterbody).geometry()
  
  jrcMonthly = ee.ImageCollection("JRC/GSW1_0/MonthlyHistory")

  def compute_area(i):
    area = ee.Image.pixelArea().mask(i.eq(2)).reduceRegion({
      'reducer': ee.Reducer.sum(), 
      'geometry': geom, 
      'scale': scale
    })

    return i.set({'area': area.get('area')})

  water = jrcMonthly.filterDate(start, stop).map(compute_area)

  return water

