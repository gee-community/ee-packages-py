import ee

from eepackages import utils

# migrated from JavaScript users/gena/packages/assets.js

def cloudMaskAlgorithms_Landsat(image):
    imageWithCloud = ee.Algorithms.Landsat.simpleCloudScore(ee.Image(image))
    
    return imageWithCloud.addBands(ee.Image(imageWithCloud.select('cloud').divide(100)), None, True)

def cloudMaskAlgorithms_S2(image):
    qa = image.select('QA60')
    #    Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    mask = mask.subtract(1).multiply(-1).rename('cloud')
    return image.addBands(mask)

cloudMaskAlgorithms = {
    'L8': cloudMaskAlgorithms_Landsat,
    'S2': cloudMaskAlgorithms_S2,
    'L7': cloudMaskAlgorithms_Landsat,
    'L5': cloudMaskAlgorithms_Landsat,
    'L4': cloudMaskAlgorithms_Landsat
}

def getImages(g, options):
    g = ee.Geometry(g)
    
    resample = False
    
    s2MergeByTime = True
    
    if options and 's2MergeByTime' in options:
        s2MergeByTime = options['s2MergeByTime']
    
    cloudMask = False
    
    if options and 'cloudMask' in options:
        cloudMask = options['cloudMask']
    
    cloudMaskAlgorithms_ = cloudMaskAlgorithms
    if options and 'cloudMaskAlgorithms' in options:
        cloudMaskAlgorithms_ = options['cloudMaskAlgorithms']
    
    missions = ['S2', 'L8']

    if options and 'missions' in options:
        missions = options['missions']
    
    if options and 'resample' in options:
        resample = options['resample']

    bands = {
        'S2': { 'from': ['B11', 'B8', 'B4', 'B3', 'B2'], 'to': ['swir', 'nir', 'red', 'green', 'blue'] },
        'L8': { 'from': ['B6', 'B5', 'B4', 'B3', 'B2'], 'to': ['swir', 'nir', 'red', 'green', 'blue'] },
        'L5': { 'from': ['B5', 'B4', 'B3', 'B2', 'B1'], 'to': ['swir', 'nir', 'red', 'green', 'blue'] },
        'L4': { 'from': ['B5', 'B4', 'B3', 'B2', 'B1'], 'to': ['swir', 'nir', 'red', 'green', 'blue'] },
        'L7': { 'from': ['B5', 'B4', 'B3', 'B2', 'B1'], 'to': ['swir', 'nir', 'red', 'green', 'blue'] },
    }

    if options and 'includeTemperature' in options:
        bands['L8']['from'].append('B10')
        bands['L8']['to'].append('temp')
        bands['L5']['from'].append('B6')
        bands['L5']['to'].append('temp')
        bands['L4']['from'].append('B6')
        bands['L4']['to'].append('temp')
        bands['L7']['from'].append('B6_VCID_1')
        bands['L7']['to'].append('temp')

    # used only for a single sensor
    if options and 'bandsAll' in options:
        bands['S2'] = { 
            'from': ['B11', 'B8', 'B3', 'B2', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B9', 'B10'], 
            'to': ['swir', 'nir', 'green', 'blue', 'red', 'red2', 'red3', 'red4', 'nir2', 'water_vapour', 'cirrus']
        }

    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(g)

    if options and 'filter' in options:
        s2 = s2.filter(options['filter'])

    # apply custom cloud masking
    if cloudMask:
        s2 = s2.map(cloudMaskAlgorithms_['S2'])
        bands['S2']['from'].append('cloud')
        bands['S2']['to'].append('cloud')

    s2 = s2.select(bands['S2']['from'], bands['S2']['to'])

    l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA')

    if options and 'includeTier2' in options:
        l8 = l8.merge(ee.ImageCollection('LANDSAT/LC08/C01/T2_TOA'))

    l8 = l8.filterBounds(g)

    if options and 'filter' in options:
        l8 = l8.filter(options['filter'])

    if cloudMask:
        l8 = l8.map(cloudMaskAlgorithms_['L8'])
        bands['L8']['from'].append('cloud')
        bands['L8']['to'].append('cloud')

    l8 = l8.select(bands['L8']['from'], bands['L8']['to'])
    
    l5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA')

    if options and 'includeTier2' in options:
        l5 = l5.merge(ee.ImageCollection('LANDSAT/LT05/C01/T2_TOA'))

    l5 = l5.filterBounds(g)

    if options and 'filter' in options:
        l5 = l5.filter(options['filter'])
    
    if cloudMask:
        l5 = l5.map(cloudMaskAlgorithms_['L5'])
        bands['L5']['from'].append('cloud')
        bands['L5']['to'].append('cloud')

    l5 = l5.select(bands['L5']['from'], bands['L5']['to'])
    
    l4 = ee.ImageCollection('LANDSAT/LT04/C01/T1_TOA')

    if options and 'includeTier2' in options:
        l4 = l4.merge(ee.ImageCollection('LANDSAT/LT04/C01/T2_TOA'))

    l4 = l4.filterBounds(g)

    if options and 'filter' in options:
        l4 = l4.filter(options['filter'])
    
    if cloudMask:
        l4 = l4.map(cloudMaskAlgorithms_['L4'])
        bands['L4']['from'].append('cloud')
        bands['L4']['to'].append('cloud')

    l4 = l4.select(bands['L4']['from'], bands['L4']['to'])

    l7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT_TOA')

    if options and 'includeTier2' in options:
        l7 = l7.merge(ee.ImageCollection('LANDSAT/LE07/C01/T2_TOA'))

    l7 = l7.filterBounds(g)

    if options and 'filter' in options:
        l7 = l7.filter(options['filter'])
    
    if cloudMask:
        l7 = l7.map(cloudMaskAlgorithms_['L7'])
        bands['L7']['from'].append('cloud')
        bands['L7']['to'].append('cloud')

    l7 = l7.select(bands['L7']['from'], bands['L7']['to'])

    if options and 'clipBufferSizeL7' in options:
        def f(i):
            mask = i.select(['green', 'red', 'nir', 'swir']).mask().reduce(ee.Reducer.allNonZero())
            mask = mask.focal_min(options['clipBufferSizeL7'], 'square', 'meters').reproject(i.select('nir').projection())
            return i.updateMask(mask)

        l7 = l7.map(f)

    clipBufferSize = 6000

    if options and 'clipBufferSize' in options:
            clipBufferSize = options['clipBufferSize']

    scale = 100
    if options and 'scale' in options:
        scale = options['scale']

    def clipNegativeFootprint(i):
        return i.clip(i.select(0).geometry().buffer(ee.Number(clipBufferSize).multiply(-1), 1000))

    l4 = l4.map(clipNegativeFootprint)
    l5 = l5.map(clipNegativeFootprint)
    l7 = l7.map(clipNegativeFootprint)

    resample = 'bicubic'

    s2 = s2.map(lambda i: i.resample(resample))

    # merge by time (remove duplicates)
    if s2MergeByTime:
        s2 = mosaicByTime(s2)

    def f2(i):
        return (i
                .addBands(i.multiply(0.0001).float(), i.bandNames().remove('cloud'), True)
                .copyProperties(i)
                .set({'SUN_ELEVATION': ee.Number(90).subtract(i.get('MEAN_SOLAR_ZENITH_ANGLE'))})
                .set({'MISSION': 'S2'})
                .set({'SUN_AZIMUTH': i.get('MEAN_SOLAR_AZIMUTH_ANGLE')})
                .set({'MULTIPLIER': 0.0001})
                )
        
    s2 = s2.map(f2)

    images = ee.ImageCollection([])

    if 'L5' in missions:
        def f(i): 
            return i.set({
                'MISSION': 'L5',
                'BANDS_FROM': bands['L5']['from'],
                'BANDS_TO': bands['L5']['to'],
                'MULTIPLIER': 1
            })

        l5 = l5.map(f)
        images = images.merge(l5)

    if 'L4' in missions:
        def f(i): 
            return i.set({
                'MISSION': 'L4',
                'BANDS_FROM': bands['L4']['from'],
                'BANDS_TO': bands['L4']['to'],
                'MULTIPLIER': 1
            })

        l4 = l4.map(f)
        images = images.merge(l4)

    if 'L7' in missions:
        def f(i): 
            return i.set({
                'MISSION': 'L7',
                'BANDS_FROM': bands['L7']['from'],
                'BANDS_TO': bands['L7']['to'],
                'MULTIPLIER': 1
            })

        l7 = l7.map(f)
        images = images.merge(l7)

    if 'L8' in missions:
        def f(i): 
            return i.set({
                'MISSION': 'L8',
                'BANDS_FROM': bands['L8']['from'],
                'BANDS_TO': bands['L8']['to'],
                'MULTIPLIER': 1
            })

        l8 = l8.map(f)
        images = images.merge(l8)

    images = ee.ImageCollection(images)
    
    if resample:
        images = images.map(lambda i: i.resample(resample))
    
    if 'S2' in missions:
        def f(i):
            return i.set({
                'MISSION': 'S2',
                'BANDS_FROM': bands['S2']['from'],
                'BANDS_TO': bands['S2']['to'],
                'MULTIPLIER': 0.0001             
            })

        s2 = s2.map(f)
        images = images.merge(s2)

    images = ee.ImageCollection(images)
    
    if options and 'filterMasked' in options:
        if 'filterMaskedFraction' in options and options['filterMaskedFraction']:
            # get images coverting bounds at least filterMaskedFraction%
            area = g.area(scale)

            def f(i):
                maskedArea = ee.Image.pixelArea().updateMask(i.select(0).mask()).reduceRegion(ee.Reducer.sum(), g, ee.Number(scale).multiply(10)).values().get(0)
                return i.set({'maskedFraction': ee.Number(maskedArea).divide(area)}) 

            images = images.map(f).filter(ee.Filter.gt('maskedFraction', options['filterMaskedFraction']))
        else:
            #    get images covering bounds 100% 
            images = images.map(lambda i: i.set({'complete': i.select(0).mask().reduceRegion(ee.Reducer.allNonZero(), g, ee.Number(scale).multiply(10)).values().get(0)})).filter(ee.Filter.eq('complete', 1))

    # exclude night images
    images = images.filter(ee.Filter.gt('SUN_ELEVATION', 0))

    return images


# /***
#  * Sentinel-2 produces multiple images, resultsing sometimes 4x more images than the actual size. 
#  * This is bad for any statistical analysis.
#  * 
#  * This function mosaics images by time.
#  */
def mosaicByTime(images):
    TIME_FIELD = 'system:time_start'

    distinct = images.distinct([TIME_FIELD])

    filter = ee.Filter.equals(**{ 'leftField': TIME_FIELD, 'rightField': TIME_FIELD });
    join = ee.Join.saveAll('matches')
    results = join.apply(distinct, images, filter)

    #    mosaic
    def merge_matches(i):
        matchedImages = ee.ImageCollection.fromImages(i.get('matches'))
        mosaic = matchedImages.sort('system:index').mosaic().set({ 'system:footprint': matchedImages.geometry() })
        
        return mosaic.copyProperties(i).set(TIME_FIELD, i.get(TIME_FIELD))
    
    results = results.map(merge_matches)
    
    return ee.ImageCollection(results)

def addQualityScore(images, g, options):
    scorePercentile = 75
    scale = 500
    mask = None
    qualityBand = 'green'

    if options:
        if 'percentile' in options:
            scorePercentile = options['percentile']
        if 'scale' in options:
            scale = options['scale']
        if 'mask' in options:
            mask = options['mask']
        if 'qualityBand' in options:
            qualityBand = options['qualityBand']

    def add_quality_score(i):
        score = i.select(qualityBand) #//.where(i.select('green').gt(0.5), 0.5)

        if mask:
            score = score.updateMask(mask)

        score = score.reduceRegion(ee.Reducer.percentile([scorePercentile]), g, scale).values().get(0)

        # // var score = i.select('green').add(i.select('blue'))
        # //    .reduceRegion(ee.Reducer.percentile([scorePercentile]), g, scale).values().get(0)

        # // var cloudScore = computeCloudScore(i)
        # // var score = cloudScore.gt(cloudThreshold)
        # //     .reduceRegion(ee.Reducer.sum(), g, scale).values().get(0)

        return i.set({ 'quality_score': score })

    return images.map(add_quality_score)

def getMostlyCleanImages(images, g, options):
    g = ee.Geometry(g)
    
    scale = 500

    if options and 'scale' in options:
        scale = options['scale']

    p = 85

    if options and 'percentile' in options:
        p = options['percentile']

    # // http://www.earthenv.org/cloud
    modisClouds = ee.Image('users/gena/MODCF_meanannual')
    
    cloudFrequency = (modisClouds.divide(10000).reduceRegion(
        ee.Reducer.percentile([p]), 
        g.buffer(10000, ee.Number(scale).multiply(10)), ee.Number(scale).multiply(10)).values().get(0))
        
    # print('Cloud frequency (over AOI):', cloudFrequency)
    
    #    decrease cloudFrequency, include some more partially-cloudy images then clip based on a quality metric
    #    also assume inter-annual variability of the cloud cover
    cloudFrequency = ee.Number(cloudFrequency).subtract(0.15).max(0.0)
    
    if options and 'cloudFrequencyThresholdDelta' in options and options['cloudFrequencyThresholdDelta']:
        cloudFrequency = cloudFrequency.add(options['cloudFrequencyThresholdDelta'])
        
    images = images.filterBounds(g)

    size = images.size()  # not being used?
    
    images = (addQualityScore(images, g, options)
        .filter(ee.Filter.gt('quality_score', 0))) # sometimes null?!

    # clip collection
    images = (images.sort('quality_score')
        .limit(images.size().multiply(ee.Number(1).subtract(cloudFrequency)).toInt()))
        
    # # remove too dark images
    # images = images.sort('quality_score', false)
    #     .limit(images.size().multiply(0.99).toInt())
    
    # print('size, filtered: ', images.size())            

    return images
        # .set({scoreMax: scoreMax})

def addCdfQualityScore(
    images,
    opt_thresholdMin=75,
    opt_thresholdMax=95,
    opt_includeNeighborhood=False,
    opt_neighborhoodOptions={"erosion": 5, "dilation": 50, "weight": 50},
    opt_qualityBand="green"
    ):
    """
    Compute one or more high percentile for every pixel and add quality score if pixel value is hither/lower than the threshold %
    """

    images = images.map(lambda i: i.addBands(i.select(opt_qualityBand).rename('q')))

    # P(bad | I < min) = 0, P(bad | I >= max) = 1
    pBad = images.select('q').reduce(ee.Reducer.percentile([opt_thresholdMin, opt_thresholdMax])).rename(['qmin', 'qmax']) 

    pBadRange = pBad.select('qmax').subtract(pBad.select('qmin'))

    def remove_bad_pixels(i):

        # probability of bad due to high reflactance values
        badLinear = i.select('q') \
            .max(pBad.select('qmin')) \
            .min(pBad.select('qmax')) \
            .subtract(pBad.select('qmin')) \
            .divide(pBadRange) \
            .clamp(0, 1)

        # too low values
        badLow = i.select('q').lt(0.0001)
            
        badWeight = badLinear.multiply(badLow.Not())

        if opt_includeNeighborhood:
            radius = opt_neighborhoodOptions

            # opening  
            bad = badWeight.gt(0.5)

            if radius.get("erosion"):
                bad = utils.focalMin(bad, radius["erosion"])

            bad = utils.focalMax(bad, radius["dilation"])

            bad = utils.focalMaxWeight(bad, radius["weight"])
            
            badWeight = badWeight.max(bad)

        # smoothen scene boundaries
        # badWeight = badWeight
        #  .multiply(utils.focalMin(i.select('blue').mask(), 10).convolve(ee.Kernel.gaussian(5, 3)))


        # compute bad pixel probability 
        weight = ee.Image(1).float().subtract(badWeight).rename('weight')

        return i.addBands(weight).addBands(pBad)

    # bad pixel removal (costly)
    images = images.map(remove_bad_pixels)

    return images

def otsu(histogram):
    """
    The script computes surface water mask using Canny Edge detector and Otsu thresholding
    See the following paper for details: http://www.mdpi.com/2072-4292/8/5/386

    Author: Gennadii Donchyts (gennadiy.donchyts@gmail.com)
    Contributors: Nicholas Clinton (nclinton@google.com) - re-implemented otsu() using ee.Array

    Example: 
        thresholding = require('users/gena/packages:thresholding')

        th = thresholding.computeThresholdUsingOtsu(image, scale, bounds, cannyThreshold, cannySigma, minValue, ...)
    
    Returns:
        the DN that maximizes interclass variance in B5 (in the region).
    """
    histogram = ee.Dictionary(histogram)

    counts = ee.Array(histogram.get('histogram'))
    means = ee.Array(histogram.get('bucketMeans'))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = sum.divide(total)

    indices = ee.List.sequence(1, size)

    #    Compute between sum of squares, where each mean partitions the data.
    def f(i):
        aCounts = counts.slice(0, 0, i)
        aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
        aMeans = means.slice(0, 0, i)
        aMean = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount)
        bCount = total.subtract(aCount)
        bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount)
        
        return aCount.multiply(aMean.subtract(mean).pow(2)).add(bCount.multiply(bMean.subtract(mean).pow(2)))

    bss = indices.map(f)

    #    Return the mean value corresponding to the maximum BSS.
    return means.sort(bss).get([-1])
