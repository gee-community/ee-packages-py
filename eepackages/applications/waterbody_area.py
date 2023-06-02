import ee
from eepackages import assets
from eepackages import utils


def computeSurfaceWaterArea(
    waterbody,
    start_filter,
    start,
    stop,
    scale,
    waterOccurrence,
    opt_missions=None,
    quality_score_attributes=None,
    mosaic_by_day=True,
    min_overlap_fraction=None,
):
    """
    compute the surface area of given waterbody between start and stop, using images between
    start_filter and stop to obtain a representative set of images to filter cloudy images.

    args:
        waterbody: waterbody feature
        start_filter: start date of images using to filter cloudy images
        start: start date of analysis
        stop: stop date of analysis
        scale: scale of analysis
        waterOccurrence: feature collection of reference water occurrence data.
            opt_missions: mission codes to use for the input imagery. Leave empty for a recommended
            set of missions.
        quality_score_attribute: if the quality_score is precomputed, a ranked list of attributes of
            the waterbody to use as quality_score. Otherwise None.
        min_overlap_fraction (Optional[float]): minimum overlap fraction between image footprint
            and waterbody. Leave as None to skip filtering.

    returns:
      image collection containing water layers.
    """

    geom = ee.Feature(waterbody).geometry()

    missions = ["L4", "L5", "L7", "L8", "S2"]
    props = waterbody.getInfo()["properties"]
    quality_score_threshold = None
    if quality_score_attributes:
        for prop in quality_score_attributes:
            att = props.get(prop)
            if att:
                quality_score_threshold = att
                break

    if opt_missions:
        missions = opt_missions

    images = assets.getImages(
        geom,
        {
            "resample": True,
            "filter": ee.Filter.date(start_filter, stop),
            "missions": missions,
            "scale": scale * 10,
        },
    )

    # TODO: mosaic by day
    # images = assets.mosaic_by_day(images)

    options = {
        # 'cloudFrequencyThresholdDelta': -0.15
        "scale": scale * 5,
        "cloud_frequency": props.get("cloud_frequency"),
        "quality_score_cloud_threshold": quality_score_threshold,
    }

    g = geom.buffer(300, scale)

    images = assets.getMostlyCleanImages(images, g, options).sort("system:time_start")
    images = images.filterDate(start, stop)

    if mosaic_by_day:
        images = assets.mosaic_by_day(images)

    if min_overlap_fraction:
        images = filter_waterbody_overlap(images, waterbody, min_overlap_fraction)

    water = images.map(
        lambda i: computeSurfaceWaterArea_SingleImage(
            i, waterbody, scale, waterOccurrence
        )
    )

    water = water.filter(ee.Filter.neq("area", 0))

    return water


def computeSurfaceWaterArea_SingleImage(i, waterbody, scale, waterOccurrence):
    geom = ee.Feature(waterbody).geometry()

    fillPercentile = 50  # // we don't trust our prior

    ndwiBands = ["green", "swir"]
    # var ndwiBands = ['green', 'nir']

    waterMaxImage = ee.Image().float().paint(waterbody.buffer(150), 1)

    maxArea = waterbody.area(scale)

    t = i.get("system:time_start")

    i = i.updateMask(waterMaxImage).updateMask(
        i.select("swir").min(i.select("nir")).gt(0.001)
    )

    ndwi = i.normalizedDifference(ndwiBands)

    # var water = ndwi.gt(0)

    th = utils.computeThresholdUsingOtsu(ndwi, scale, geom, 0.5, 0.7, -0.2)
    water = ndwi.gt(th)

    area = (
        ee.Image.pixelArea()
        .mask(water)
        .reduceRegion(**{"reducer": ee.Reducer.sum(), "geometry": geom, "scale": scale})
        .get("area")
    )

    # # fill missing, estimate water probability as the lowest percentile.
    # var waterEdge = ee.Algorithms.CannyEdgeDetector(water, 0.1, 0)
    waterEdge = ee.Algorithms.CannyEdgeDetector(ndwi, 0.5, 0.7)

    # image mask
    imageMask = ndwi.mask()

    # var imageMask = i.select(ndwiBands).reduce(ee.Reducer.allNonZero())

    # imageMask = utils.focalMin(imageMask, ee.Number(scale) * 1.5)
    imageMask = imageMask.focal_min(ee.Number(scale).multiply(1.5), "square", "meters")
    # .multiply(waterMaxImage)

    # TODO: exclude non-water/missing boundsry
    waterEdge = waterEdge.updateMask(imageMask)

    # # clip by clouds
    # var bad = ee.Image(1).float().subtract(i.select('weight'))
    # bad = utils.focalMax(bad, 90).not()
    # waterEdge = waterEdge.updateMask(bad)

    # get water probability around edges
    # P(D|W) = P(D|W) * P(W) / P(D) ~=  P(D|W) * P(W)
    p = (
        waterOccurrence.mask(waterEdge)
        .reduceRegion(
            **{
                "reducer": ee.Reducer.percentile([fillPercentile]),
                "geometry": geom,
                "scale": scale,
            }
        )
        .values()
        .get(0)
    )

    # TODO: exclude edges belonging to cloud/water or cloud/land

    # TODO: use multiple percentiles (confidence margin)

    p = ee.Algorithms.If(ee.Algorithms.IsEqual(p, None), 101, p)

    waterFill = waterOccurrence.gt(ee.Image.constant(p)).updateMask(
        water.unmask(0, False).Not()
    )

    # exclude false-positive, where we're sure in a non-water
    nonWater = ndwi.lt(-0.15).unmask(0, False)
    waterFill = waterFill.updateMask(nonWater.Not())

    fill = (
        ee.Image.pixelArea()
        .mask(waterFill)
        .reduceRegion(**{"reducer": ee.Reducer.sum(), "geometry": geom, "scale": scale})
        .get("area")
    )

    area_filled = ee.Number(area).add(fill)

    filled_fraction = ee.Number(fill).divide(area_filled)

    return (
        i.addBands(waterFill.rename("water_fill"))
        .addBands(waterEdge.rename("water_edge"))
        .addBands(ndwi.rename("ndwi"))
        .addBands(water.rename("water"))
        .set(
            {
                "p": p,
                "area": area,
                "area_filled": area_filled,
                "filled_fraction": filled_fraction,
                "system:time_start": t,
                "ndwi_threshold": th,
                "waterOccurrenceExpected": waterOccurrence.mask(waterEdge),
            }
        )
    )


def computeSurfaceWaterAreaJRC(waterbody, start, stop, scale):
    geom = ee.Feature(waterbody).geometry()

    jrcMonthly = ee.ImageCollection("JRC/GSW1_0/MonthlyHistory")

    def compute_area(i):
        area = (
            ee.Image.pixelArea()
            .mask(i.eq(2))
            .reduceRegion(
                {"reducer": ee.Reducer.sum(), "geometry": geom, "scale": scale}
            )
        )

        return i.set({"area": area.get("area")})

    water = jrcMonthly.filterDate(start, stop).map(compute_area)

    return water


def filter_waterbody_overlap(images, waterbody, min_overlap_fraction=0.4):
    """
    filter images based on the overlap of the image footprint with the waterbody.
    If the overlap consists of less than `min_overlap_fraction` of the waterbody area, the image
    if filtered out.

    Args:
        images: ImageCollection of input images
        waterbody: Feature with waterbody
        min_overlap_fraction: minumum fraction of overlap to accept.
    returns:
        filtered ImageCollection
    """

    def set_intersection(i):
        intersection_area = (
            ee.Geometry(i.get("system:footprint"))
            .intersection(waterbody.geometry())
            .area()
        )
        waterbody_area = waterbody.geometry().area()
        i = i.set("overlap_fraction", intersection_area.divide(waterbody_area))
        return i

    images = images.map(set_intersection)
    images = images.filter(ee.Filter.gte("overlap_fraction", min_overlap_fraction))

    return images


def extrapolate_JRC_wo(waterbody, scale, max_trusted_occurrence=0.97):
    """
    extrapolate the JRC dataset based on a maximum value that is trusted.
    This is needed when a waterbody experiences an unprecedented drought, in which gap filling
    fails when using the JRC dataset as-is.
    """
    water_occurrence = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .mask(1)  # fixes JRC masking lacking
        .resample("bicubic")
        .divide(100)
    )

    # we don't trust water occurrence dataset when water level is below this (values > th)
    min_water = water_occurrence.gt(max_trusted_occurrence)

    # Calculate distance from water Occurence max
    dist = (
        min_water.Not()
        .fastDistanceTransform(150)
        .reproject(ee.Projection("EPSG:4326").atScale(100))
        .resample("bicubic")
        .sqrt()
        .multiply(100)
    )

    # calculate max distance for scaling to 1
    max_distance = dist.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=waterbody.geometry(),
        scale=scale,
        bestEffort=True,
    ).get("distance")

    # scale distance values from min_trusted_occurrence to 1
    extrap_scale = ee.Number(1).subtract(max_trusted_occurrence).divide(max_distance)
    water_occurrence_extrapolated = dist.multiply(extrap_scale).add(
        max_trusted_occurrence
    )

    return water_occurrence.where(
        water_occurrence.gt(max_trusted_occurrence), water_occurrence_extrapolated
    )
