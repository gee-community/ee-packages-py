from typing import Any, Dict, List, Optional
from datetime import datetime

import ee

from eepackages import assets
from eepackages import gl
from eepackages import utils

# TODO: Jaap to clean up this script following the same style as original product

# load GTSM & gebco data
gtsm_col = ee.FeatureCollection("projects/bathymetry/assets/gtsm_waterlevels")
gebco_image = ee.Image("projects/bathymetry/assets/gebco_2023_hat_lat")


class Bathymetry(object):
    def __init__(self, waterIndexMin: float = -0.15, waterIndexMax: float = 0.35):
        self.waterIndexMin = waterIndexMin
        self.waterIndexMax = waterIndexMax

    @staticmethod
    def _remove_all_zero_images(image: ee.Image):
        mask: ee.Image = (
            image.select(["blue", "green", "red", "nir", "swir"])
            .mask()
            .reduce(ee.Reducer.allNonZero())
        )
        return image.updateMask(mask)

    def compute_inverse_depth(
        self,
        bounds,
        start: datetime,
        stop: datetime,
        filter_masked: bool,
        scale: float,
        missions: List[str] = ["S2", "L8"],
        cloud_frequency_threshold_data: float = 0.15,
        pansharpen: bool = False,
        skip_neighborhood_search: bool = False,
        skip_scene_boundary_fix: bool = False,
        bounds_buffer: int = 10000,
    ) -> ee.Image:
        images: ee.ImageCollection = self.get_images(
            bounds=bounds,
            start=start,
            stop=stop,
            filter_masked=filter_masked,
            scale=scale,
            missions=missions,
            cloud_frequency_threshold_delta=cloud_frequency_threshold_data,
        )
        # save loaded images in class as raw_images
        self._raw_images = images
        images = images.map(self._remove_all_zero_images)

        bounds = bounds.buffer(bounds_buffer, ee.Number(bounds_buffer).divide(10))

        return self._compute_inverse_depth(
            images=images,
            bounds=bounds,
            scale=scale,
            pansharpen=pansharpen,
            skip_neighborhood_search=skip_neighborhood_search,
            skip_scene_boundary_fix=skip_scene_boundary_fix,
        )

    def _compute_inverse_depth(
        self,
        images: ee.ImageCollection,
        bounds,
        scale: int,
        pansharpen: bool,
        skip_neighborhood_search: bool,
        skip_scene_boundary_fix: bool,
    ) -> ee.Image:
        bands: List[str] = ["red", "green", "blue"]
        green_max: float = 0.4

        def _set_image_area_properties(image: ee.Image) -> ee.Image:
            water: ee.Image = (
                image.normalizedDifference(["green", "nir"])
                .rename("water")
                .unitScale(0, 0.1)
            )

            if pansharpen:
                # TODO: Not implemented
                raise NotImplementedError()
                image = assets.pansharpen(image)

            water_area: ee.Number = (
                water.gt(0.01)
                .multiply(ee.Image.pixelArea())
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=bounds,
                    scale=ee.Number(scale).multiply(5),
                    tileScale=4,
                )
                .values()
                .get(0)
            )

            land_area: ee.Number = (
                water.lt(0)
                .multiply(ee.Image.pixelArea())
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=bounds,
                    scale=ee.Number(scale).multiply(5),
                    tileScale=4,
                )
                .values()
                .get(0)
            )

            dark: ee.Image = image

            dark = dark.updateMask(water.gt(0)).reduceRegion(
                reducer=ee.Reducer.percentile([0]),
                geometry=bounds,
                scale=scale,
                maxPixels=1e10,
                tileScale=4,
            )

            image = image.set(dark).set(
                {"water": water, "waterArea": water_area, "landArea": land_area}
            )

            return image

        images: ee.ImageCollection = images.map(_set_image_area_properties)
        self._images_area_properties = images

        # Filter images with negative RGB values
        images = images.filter(
            ee.Filter.And(
                ee.Filter.gt(bands[0], 0),
                ee.Filter.gt(bands[1], 0),
                ee.Filter.gt(bands[2], 0),
            )
        )

        if skip_neighborhood_search:
            images = assets.addCdfQualityScore(
                images=images,
                opt_thresholdMin=70,
                opt_thresholdMax=80,
                opt_includeNeighborhood=False,
            )
        else:
            images = assets.addCdfQualityScore(
                images=images,
                opt_thresholdMin=70,
                opt_thresholdMax=80,
                opt_includeNeighborhood=True,
                opt_neighborhoodOptions={"erosion": 0, "dilation": 0, "weight": 200},
            )

        def fix_scene_boundaries(image: ee.Image) -> ee.Image:
            weight: ee.Image = image.select("weight")
            mask = image.select(0).mask()

            mask: ee.Image = (
                utils.focalMin(mask, 10)
                .reproject(ee.Projection("EPSG:3857").atScale(scale))
                .resample("bicubic")
            )
            mask = (
                utils.focalMaxWeight(mask.Not(), 10)
                .reproject(ee.Projection("EPSG:3857").atScale(scale))
                .resample("bicubic")
            )
            mask = ee.Image.constant(1).subtract(mask)

            weight = weight.multiply(mask)

            return image.addBands(weight, None, True)

        if not skip_scene_boundary_fix:
            images = images.map(fix_scene_boundaries)

        def image_map_func(i: ee.Image):
            t: str = i.get("system:time_start")
            weight: ee.Image = i.select("weight")

            dark_image: ee.Image = ee.Image.constant(
                list(map(lambda n: i.get(n), bands))
            ).rename(bands)
            mission: str = i.get("MISSION")
            scale_water_to: str = "percentiles"
            scale_land_to: str = "percentiles"

            range_percentiles_water: List[int] = [2, 98]
            range_percentiles_land: List[int] = [2, 98]

            range_sigma_water: List[int] = [1, 1]
            range_sigma_land: List[int] = [2, 2]

            water: ee.Image = ee.Image(i.get("water"))
            non_water: ee.Image = water.subtract(1).multiply(-1)  # Not being used

            i = i.select(bands).subtract(dark_image).max(0.0001)

            i_all: ee.Image = i

            water2: ee.Image = gl.smoothStep(-0.05, 0.2, water)
            non_water2: ee.Image = water2.subtract(1).multiply(-1)

            i = i.log()

            stat1: ee.Image = i

            stat1 = stat1.updateMask(
                water2.multiply(i_all.select("green").lt(green_max))
            )

            if scale_water_to == "percentiles":
                stat1 = stat1.reduceRegion(
                    reducer=ee.Reducer.percentile(range_percentiles_water),
                    geometry=bounds,
                    scale=ee.Number(scale).multiply(3),
                    maxPixels=1e10,
                )

                min1: List[str] = [
                    ee.String(stat1.get(f"{band}_p{range_percentiles_water[0]}"))
                    for band in bands
                ]
                max1: List[str] = [
                    ee.String(stat1.get(f"{band}_p{range_percentiles_water[1]}"))
                    for band in bands
                ]

            if scale_water_to == "sigma":
                stat1mean = stat1.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=bounds,
                    scale=ee.Number(scale).multiply(3),
                    maxPixels=1e10,
                )

                stat1sigma = stat1.reduceregion(  # not being used.
                    reducer=ee.Reducer.stDev(),
                    geometry=bounds,
                    scale=ee.Number(scale).multiply(3),
                    maxPixels=1e10,
                )

                # Not sure whether this si tested, min1 should always be zero
                min1: List[ee.Number] = [
                    ee.Number(stat1mean.get(band)).subtract(
                        ee.Number(stat1mean.get(band)).multiply(range_sigma_water[0])
                    )
                    for band in bands
                ]
                max1: List[ee.Number] = [
                    ee.Number(stat1mean.get(band)).add(
                        ee.Number(stat1mean.get(band)).multiply(range_sigma_water[1])
                    )
                    for band in bands
                ]

            min1 = self._fix_null(min1, 0)
            max1 = self._fix_null(max1, 0.001)

            stat2: ee.Image = i_all.updateMask(
                non_water2.multiply(i_all.select("green").lt(green_max))
            )

            if scale_land_to == "percentiles":
                stat2 = stat2.reduceRegion(
                    reducer=ee.Reducer.percentile(range_percentiles_land),
                    geometry=bounds,
                    scale=ee.Number(scale).multiply(3),
                    maxPixels=1e10,
                )

                min2 = [
                    ee.String(stat2.get(f"{band}_p{range_percentiles_land[0]}"))
                    for band in bands
                ]
                max2 = [
                    ee.String(stat2.get(f"{band}_p{range_percentiles_land[1]}"))
                    for band in bands
                ]

            if scale_land_to == "sigma":
                stat2mean = stat2.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=bounds,
                    scale=ee.Number(scale).multiply(3),
                    maxPixels=1e10,
                )
                stat2sigma = stat2.reduceRegion(  # not used?
                    reducer=ee.Reducer.stDev(),
                    geometry=bounds,
                    scale=ee.Number(scale).multiply(3),
                    maxPixels=1e10,
                )

                min2: List[ee.Number] = [
                    ee.Number(stat2mean)
                    .get(band)
                    .subtract(
                        ee.Number(stat2mean.get(band)).multiply(range_sigma_land[0])
                    )
                    for band in bands
                ]
                max2: List[ee.Number] = [
                    ee.Number(stat2mean.get(band)).add(
                        ee.Number(stat2mean.get(band)).multiply(range_sigma_land[1])
                    )
                    for band in bands
                ]

            min2 = self._fix_null(min2, 0)
            max2 = self._fix_null(max2, 0.001)

            i_water = self._unit_scale(i.select(bands), min1, max1).updateMask(water2)

            i_land = self._unit_scale(i_all.select(bands), min2, max2).updateMask(
                non_water2
            )

            i = i_water.blend(i_land).addBands(water)

            i = i.addBands(weight)

            return i.set(
                {
                    "label": ee.Date(t).format().cat(", ").cat(mission),
                    "system:time_start": t,
                }
            )

        images = images.map(image_map_func)

        self._images_with_statistics = images

        # mean = sum(w * x) / sum(w)
        image: ee.Image = (
            images.map(
                lambda i: i.select(bands + ["water"]).multiply(i.select("weight"))
            )
            .sum()
            .divide(images.select("weight").sum())
        )

        return image

    def compute_intertidal_depth(
        self,
        bounds,
        start: datetime,
        stop: datetime,
        scale: float,
        filter_masked: bool,
        filter_masked_fraction: Optional[float] = None,
        filter: Optional[ee.Filter] = None,
        bounds_buffer: Optional[float] = None,
        water_index_min: Optional[float] = None,
        water_index_max: Optional[float] = None,
        missions: List[str] = ["S2", "L8"],
        skip_scene_boundary_fix=False,
        skip_neighborhood_search: bool = False,
        neighborhood_search_parameters: Dict[str, float] = {
            "erosion": 0,
            "dilation": 0,
            "weight": 100,
        },
        lower_cdf_boundary: float = 70,
        upper_cdf_boundary: float = 80,
        cloud_frequency_threshold_data: float = 0.15,
        clip: bool = False,
    ) -> ee.Image:
        if water_index_min:
            self.waterIndexMin = water_index_min
        water_index_min = self.waterIndexMin
        if water_index_max:
            self.waterIndexMax = water_index_max
        water_index_max = self.waterIndexMax
        if bounds_buffer:
            bounds = bounds.buffer(bounds_buffer, bounds_buffer / 10)

        images: ee.ImageCollection = self.get_images(
            bounds=bounds,
            start=start,
            stop=stop,
            filter_masked=filter_masked,
            filter_masked_fraction=filter_masked_fraction,
            scale=scale,
            missions=missions,
            cloud_frequency_threshold_delta=cloud_frequency_threshold_data,
            filter=filter,
        )

        self._raw_images = images

        bands: List[str] = ["blue", "green", "red", "nir", "swir"]

        # Mask all zero images
        def mask_zero_images(i: ee.Image) -> ee.Image:
            mask: ee.Image = (
                i.select(bands).mask().reduce(ee.Reducer.allNonZero()).eq(1)
            )
            return i.updateMask(mask)

        images = images.map(mask_zero_images)

        if not bounds_buffer:
            bounds_buffer = 10000  # Not being used.

        if skip_neighborhood_search:
            images = assets.addCdfQualityScore(images, 70, 80, False)
        else:
            images = assets.addCdfQualityScore(
                images=images,
                opt_thresholdMin=lower_cdf_boundary,
                opt_thresholdMax=upper_cdf_boundary,
                opt_includeNeighborhood=True,
                opt_neighborhoodOptions=neighborhood_search_parameters,
            )

        if not skip_scene_boundary_fix:

            def fix_scene_boundaries(i: ee.Image) -> ee.Image:
                weight: ee.Image = i.select("weight")
                mask: ee.Image = i.select(0).mask()

                mask = (
                    utils.focalMin(mask, 10)
                    .reproject(ee.Projection("EPSG:3857").atScale(scale))
                    .resample("bicubic")
                )
                mask = (
                    utils.focalMaxWeight(mask.Not(), 10)
                    .reproject(ee.Projection("EPSG:3857").atScale(scale))
                    .resample("bicubic")
                )
                mask = ee.Image.constant(1).subtract(mask)

                weight = weight.multiply(mask)

                return i.addBands(srcImg=weight, overwrite=True)

            images = images.map(fix_scene_boundaries)

        self._refined_images = images

        # ADD-INS (method improvements)

        # map GTSM & GEBCO on the image collection
        GTSMcol = images.map(lambda image: self.add_gtsm_gebco_data_to_images(image.clip(bounds), gtsm_col, ee.Feature(bounds))) # TODO: test.clip(bounds)

        filteredGTSM = GTSMcol.filter(ee.Filter.notNull(['gtsm_feature'])) # images with matching GTSM data
        filteredNoGTSM = GTSMcol.filter(ee.Filter.notNull(['gtsm_feature']).Not()) # images without matching GTSM data

        # map collection to set image properties
        filteredGTSM = filteredGTSM.map(lambda image: self.set_gtsm_gebco_data_to_images(image, gebco_image))
        filteredNoGTSM = filteredNoGTSM.map(lambda image: image.set({'gtsm_gebco_data_isempty': True})) #True

        self._images_WLinfo = filteredGTSM

        # Below comes a complex situation because we want to use getInfo & ee.Algorithms.if as little as possible so we need to work with map & filters, yet, we cannot succeed to get rid of all
        # the ee.Algorithms.if because we have 3 options; all data in filteredGTSM, data in both filteredGTSM & filteredNoGTSM and data in only filteredNoGTSM. In case of the former two, 
        # get_tide_offsets_and_spread and calibrated_bathy get data with GTSM info coupled and everything works fine. In case of the latter, it will break without if statement because these 
        # functions cannot handle zero imagecollections.. The crux is really in the get_tide_offsets_and_spread. We cannot combine the two imagecollections before because it will break on data 
        # without GTSM info. We also cannot combine after because it will break on an empty filteredGTSM list (latter option). Besides, we want to keep tehm seperate to calibrate the filteredGTSM
        # collection if we have it.. TODO: see if we can get rid of ee.algorithms.if after all..  
        # ideas: https://gis.stackexchange.com/questions/414324/handling-null-images-using-map-function-with-google-earth-engine-python-api; refactor complete functionality by giving back
        # compute_intertidal_depth after step above. Then filter out the tile already that has no GTSM coupled and then continue with the calibration of tiles with GTSM. 

        # compute bool_empty ImageCollection is empty
        bool_empty_filGTSM = filteredGTSM.size().eq(0)
        bool_empty_filNoGTSM = filteredNoGTSM.size().eq(0)

        # Use two server-side conditional statements to keep memory usage low by comparing against an empty imagecollection as both true and false conditions are calculated at once. 
        # See: https://developers.google.com/earth-engine/apidocs/ee-algorithms-if
        image_calib = ee.Image(ee.Algorithms.If(bool_empty_filGTSM, ee.ImageCollection([]).first(), self.compute_bathy_GTSM(filteredGTSM)))
        image_uncalib = ee.Image(ee.Algorithms.If(bool_empty_filNoGTSM, ee.ImageCollection([]).first(), self.compute_proxy_NoGTSM(filteredNoGTSM)))

        # merge the images
        image_bp = ee.ImageCollection([image_calib, image_uncalib])
        image = image_bp.first() #this select the first image; image_bathy if it exists, else image_proxy

        # END ADD-INS

        # mean = sum(w * x) / sum (w)
        # self.composite = (
        #     images.map(lambda i: i.select(bands).multiply(i.select("weight")))
        #     .sum()
        #     .divide(images.select("weight").sum())
        #     .select(["red", "green", "blue", "swir", "nir"])
        # )

        # bands = ["ndwi", "indvi", "mndwi"]

        # def calculate_water_statistics(i: ee.Image) -> ee.Image:
        #     t = i.get("system:time_start")  # Not used
        #     weight: ee.Image = i.select("weight")

        #     ndwi: ee.Image = i.normalizedDifference(["green", "nir"]).rename("ndwi")
        #     indvi: ee.Image = i.normalizedDifference(["red", "nir"]).rename("indvi")
        #     mndwi: ee.Image = i.normalizedDifference(["green", "swir"]).rename("mndwi")

        #     ndwi = ndwi.clamp(water_index_min, water_index_max)
        #     indvi = indvi.clamp(water_index_min, water_index_max)
        #     mndwi = mndwi.clamp(water_index_min, water_index_max)

        #     return ee.Image([ndwi, indvi, mndwi]).addBands(weight)

        # images = images.map(calculate_water_statistics)

        # self._images_with_statistics = images

        # image = (
        #     images.map(lambda i: i.select(bands).multiply(i.select("weight")))
        #     .sum()
        #     .divide(images.select("weight").sum())
        # )

        if clip == True:
            return image.clip(bounds)
        else:
            return image

    # Add gtsm and gebco data to images
    @staticmethod
    def add_gtsm_gebco_data_to_images(image, gtsm_col, tile=ee.Feature(None),
                                    max_spatial_offset=1, max_temporal_offset=10):
        ''' Add gtsm and gebco data to images.

        :param image: Image to which gtsm data is added.
        :type image: ee.Image
        :param gtsm_col: gtsm feature collection with water levels.
        :type gtsm_col: ee.FeatureCollection
        :param tile: Tile to which image belongs (default=None, tile geometry is determined based on image geometry).
        :type tile: ee.Feature
        :param max_spatial_offset: Maximum spatial offset in kilometers.
        :type max_spatial_offset: float (default=1)
        :param max_temporal_offset: Maximum temporal offset in minutes
        :type max_temporal_offset: float (default=10)
        '''
        
        # If tile geometry is not provided, set tile geometry to image geometry
        # tile = ee.Feature(ee.Algorithms.If(ee.Algorithms.IsEqual(tile.geometry(), None),
        # 								   ee.Feature(image.geometry()),
        # 								   tile)) # NOTE, commented out because we always map over a tile?

        # Get area around the tile
        tile_centroid = ee.Geometry.centroid(tile.geometry(), maxError=1)
        tile_footprint = ee.Geometry(tile.geometry())
        tile_buffer = tile_footprint.buffer(max_spatial_offset*1000)

        # Get period around image time
        image_time_start = ee.Date(image.get('system:time_start'))
        # image_time_end = ee.Date(ee.Algorithms.If(image.get('system:time_end'),
        # 										  ee.Date(image.get('system:time_end')),
        # 										  image_time_start))
        image_time_end = ee.Date(image.get('system:time_start')) # TODO: fix nicely with time_end.. 
        image_time = ee.Date(image_time_start.millis().add(image_time_end.millis()).divide(2))
        image_period = ee.DateRange(ee.Date(image_time_start.millis().subtract(max_temporal_offset*60*1000)),
                                    ee.Date(image_time_end.millis().add(max_temporal_offset*60*1000)))
        
        # Filter gtsm data based on image footprint and period
        gtsm_col = gtsm_col.filterBounds(tile_buffer)
        gtsm_col = gtsm_col.filterDate(image_period.start(), image_period.end())

        # Add spatial offset to features
        def add_spatial_offset_to_features(feature):
            return feature.set('spatial offset to image', feature.distance(ee.Feature(tile_centroid)))
        gtsm_col = gtsm_col.map(add_spatial_offset_to_features)

        # Get minimum spatial offset
        min_spatial_offset = gtsm_col.reduceColumns(ee.Reducer.min(), ['spatial offset to image']).get('min')
        
        # Get features for which the spatial offset is equal to the minimum spatial offset (multiple features possible)
        gtsm_col = gtsm_col.filter(ee.Filter.eq('spatial offset to image', min_spatial_offset))

        # Add temporal offset to features
        def add_temporal_offset_to_features(feature):
            return feature.set('temporal offset to image', ee.Number(feature.get('system:time_start')).subtract(image_time.millis()).abs())
        gtsm_col = gtsm_col.map(add_temporal_offset_to_features)

        # Get minimum temporal offset
        min_temporal_offset = gtsm_col.reduceColumns(ee.Reducer.min(), ['temporal offset to image']).get('min')

        # Get features for which the temporal offset is equal to the minimum temporal offset (multiple features possible)
        gtsm_col = gtsm_col.filter(ee.Filter.eq('temporal offset to image', min_temporal_offset))

        # Add GTSM feature to image
        image = image.set("gtsm_feature", gtsm_col.first())

        return image

    # Set gtsm and gebco data to images
    # TODO: Ruben gtsm_station_lon & lat does not work..
    @staticmethod
    def set_gtsm_gebco_data_to_images(image, gebco_image, max_spatial_offset=1): 
        ''' Add gtsm and gebco data to images.

        :param image: Image to which gtsm data is added.
        :type image: ee.Image
        :param gebco_image: gebco image with highest and lowest astronomical tide.
        :type gebco_image: ee.Image
        :param max_spatial_offset: Maximum spatial offset in kilometers.
        :type max_spatial_offset: float (default=1)
        '''

        # Get gtsm feature
        gtsm_feature = ee.Feature(image.get('gtsm_feature'))

        # Get station buffer
        station_buffer = ee.Geometry(gtsm_feature.geometry().buffer(max_spatial_offset*1000))
        
        # Get gebco highest and lowest astronomical tide data
        gebco_data = ee.Dictionary(gebco_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=station_buffer, scale=30))
        
        # Get gtsm tidal stage percentage: (WL - LAT) / (HAT - LAT) * 100
        def get_gtsm_tidal_stage_percentage(gtsm_feature, gebco_data):
            wl = ee.Number(gtsm_feature.get('waterlevel'))
            lat = ee.Number(gebco_data.get('b2'))
            hat = ee.Number(gebco_data.get('b1'))
            return wl.subtract(lat).divide(hat.subtract(lat)).multiply(100)
        
        gtsm_tidal_stage_percentage = ee.Number(get_gtsm_tidal_stage_percentage(gtsm_feature, gebco_data))
        
        # Set gtsm en gebco data to image
        image = image.set({'gtsm_gebco_data_isempty': False, # False
                            'gtsm_station': gtsm_feature.get('station'),
                            'gtsm_station_lon': gtsm_feature.get('lon'),
                            'gtsm_station_lat': gtsm_feature.get('lat'),
                            'gtsm_time': gtsm_feature.get('times'),
                            'gtsm_waterlevel': gtsm_feature.get('waterlevel'),
                            'gebco_hat': gebco_data.get('b1'),
                            'gebco_lat': gebco_data.get('b2'),
                            'gtsm_tidal_stage_percentage': gtsm_tidal_stage_percentage})
        
        # Return image
        return image

    # https://developers.google.com/earth-engine/guides/ic_mapping
    # Overall function to compute bathy with GTSM data
    @staticmethod
    def compute_bathy_GTSM(image_col):
        # setter
        filteredGTSM = image_col

        # Get high tide offset, low tide offset and tide spread
        # got rid of all ee.Algroithms.Ifs with filters: https://gis.stackexchange.com/questions/478868/update-featurecollection-property-values-based-on-condition-without-using-ee-al 
        def get_tide_offsets_and_spread(image_col):
            ''' Get high tide offset, low tide offset and tide spread.

            :param image_col: Image collection with gtsm and gebco data.
            :type image_col: ee.ImageCollection
            '''

            # Filter images with gtsm and gebco data
            #image_col_ = image_col.filterMetadata('gtsm_gebco_data_isempty', 'equals', False)

            # Get high tide offset & max water level
            high_tide_offset = ee.Number(ee.Number(100).subtract(ee.Number(image_col.reduceColumns(ee.Reducer.max(), ['gtsm_tidal_stage_percentage']).get('max'))))
            max_wl = ee.Number(image_col.reduceColumns(ee.Reducer.max(), ['gtsm_waterlevel']).get('max'))

            # Get low tide offset & min water level
            low_tide_offset = ee.Number(image_col.reduceColumns(ee.Reducer.min(), ['gtsm_tidal_stage_percentage']).get('min'))
            min_wl = ee.Number(image_col.reduceColumns(ee.Reducer.min(), ['gtsm_waterlevel']).get('min'))

            # Get tide spread
            tide_spread = ee.Number(ee.Number(100).subtract(high_tide_offset).subtract(low_tide_offset))

            # Add high tide offset, low tide offset and tide spread to image collection properties
            image_col = image_col.set({'gtsm_gebco_data_allempty': False,
                                        'high_tide_offset': high_tide_offset,
                                        'low_tide_offset': low_tide_offset,
                                        'max_water_level': max_wl,
                                        'min_water_level': min_wl,
                                        'tide_spread': tide_spread})
                
            return image_col, high_tide_offset, low_tide_offset, max_wl, min_wl, tide_spread

        # get tide offsets and spread
        filteredGTSM, HTO, LTO, MAWL, MIWL, SPR = get_tide_offsets_and_spread(filteredGTSM) # gives an error for empty filteredGTSM collections... might merge into calibrated bathy

        # get depth proxy
        gridCellWaterOccurrenceOutput, NDWICollectionGTSMMapped = assets.depth_proxy(filteredGTSM)

        # Calculate the calibrated bathymetry
        def calibrated_bathy(image_col, NDWI_col, HTO, LTO, MAWL, MIWL, SPR):

            gridCellWaterOccurrenceOutput = image_col
            NDWICollectionGTSMMapped = NDWI_col

            # intertidal elevation and tidal stage (couple all)
            # short-cut to produce only the linear wlmax-wlmin scaled image to get the tidally corrected output with the overal (median or) mean NWDI image (water occurrence)
            # TODO: check the effect of this compared to the tidal stage intervals output.. (non-linear)
            waterElev = gridCellWaterOccurrenceOutput.select("waterOccurrencePercentage")\
                        .unitScale(0, 100)\
                        .multiply(ee.Number(NDWICollectionGTSMMapped.get("min_water_level")).subtract(ee.Number(NDWICollectionGTSMMapped.get("max_water_level"))))\
                        .add(ee.Number(NDWICollectionGTSMMapped.get("max_water_level")))\
                        .rename("intertidal_elevation")
            gridCellWaterOccurrenceOutput = gridCellWaterOccurrenceOutput.addBands(waterElev) #add intertidal elevation as a band to the water occurrence image
            waterElevWeight = gridCellWaterOccurrenceOutput.select("waterOccurrencePercentageWeighted")\
                        .unitScale(0, 100)\
                        .multiply(ee.Number(NDWICollectionGTSMMapped.get("min_water_level")).subtract(ee.Number(NDWICollectionGTSMMapped.get("max_water_level"))))\
                        .add(ee.Number(NDWICollectionGTSMMapped.get("max_water_level")))\
                        .rename("intertidal_elevationweighted")
            gridCellWaterOccurrenceOutput = gridCellWaterOccurrenceOutput.addBands(waterElevWeight) #add weighted intertidal elevation as a band to the water occurrence image

            # Add high tide offset, low tide offset and tide spread to image collection properties
            out_img = gridCellWaterOccurrenceOutput.set({'gtsm_gebco_data_allempty': False,
                                                                            'high_tide_offset': HTO,
                                                                            'low_tide_offset': LTO,
                                                                            'max_water_level': MAWL,
                                                                            'min_water_level': MIWL,
                                                                            'tide_spread': SPR})

            # select image to export, either water occurrence percentage (proxy, no calibrated) or depth (calibrated)
            image_bathy = ee.Image(out_img.select("intertidal_elevationweighted"))

            return image_bathy

        # calibrate the depth proxy
        image_bathy = calibrated_bathy(gridCellWaterOccurrenceOutput, NDWICollectionGTSMMapped, HTO, LTO, MAWL, MIWL, SPR)

        return image_bathy

    # Overall function to compute bathy without GTSM data
    @staticmethod
    def compute_proxy_NoGTSM(image_col):
        # setter
        filteredNoGTSM = image_col

        # get tide offsets and spread 
        filteredNoGTSM = filteredNoGTSM.set({'gtsm_gebco_data_allempty': True})

        # get depth proxy
        gridCellWaterOccurrenceOutput, NDWICollectionGTSMMapped = assets.depth_proxy(filteredNoGTSM)

        # Calculate the uncalibrated bathymetry
        def uncalibrated_bathy(image_col):

            gridCellWaterOccurrenceOutput = image_col

            # final output image, depends if GTSM data was found if this has depths or only proxies (the latter wont be exported later..)
            out_img = ee.Image(gridCellWaterOccurrenceOutput.set({"gtsm_gebco_data_allempty": True}))
            
            # # select image to export, either water occurrence percentage (proxy, no calibrated) or depth (calibrated)
            image_bathy = ee.Image(out_img.select("waterOccurrencePercentageWeighted"))

            return image_bathy

        # uncalibrated bathy
        image_bathy = uncalibrated_bathy(gridCellWaterOccurrenceOutput)
        
        return image_bathy

    @staticmethod
    def get_images(
        bounds,
        start: datetime,
        stop: datetime,
        filter_masked: bool,
        scale: float,
        missions: List[str],
        filter_masked_fraction: Optional[float] = None,
        cloud_frequency_threshold_delta: float = 0.15,
        filter: Optional[ee.Filter] = None,
    ) -> ee.ImageCollection:
        date_filter: ee.Filter = ee.Filter.date(start, stop)
        if filter:
            filter: ee.Filter = ee.Filter.And(filter, date_filter)
        else:
            filter: ee.Filter = date_filter

        options_get_images: Dict[str, Any] = {
            "missions": missions,
            "filter": filter,
            "filterMasked": filter_masked,
            "filterMaskedFraction": filter_masked_fraction,
            "scale": ee.Number(scale).multiply(10),  # why *10?
            "resample": True,
        }

        images: ee.ImageCollection = assets.getImages(bounds, options_get_images)

        options_get_mostly_clean_images: Dict[str, Any] = {
            "cloudFrequencyThresholdDelta": cloud_frequency_threshold_delta
        }

        return assets.getMostlyCleanImages(
            images, bounds, options_get_mostly_clean_images
        )

    @staticmethod
    def _fix_null(values, v) -> ee.List:
        return ee.List(values).map(
            lambda o: ee.Algorithms.If(ee.Algorithms.IsEqual(o, None), v, o)
        )

    @staticmethod
    def _unit_scale(image: ee.Image, min: float, max: float) -> ee.Image:
        min_image: ee.Image = ee.Image.constant(min)
        max_image: ee.Image = ee.Image.constant(max)
        return image.subtract(min_image).divide(max_image.subtract(min_image))
