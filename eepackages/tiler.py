import math
from typing import Optional

import ee

"""
creating a tiler using earthengine classes, so it can be used on-cluster in a parallel way.
"""

TILE_SIZE: float = 256
ORIGIN: float = math.pi * 6378137  # earth depth
C: float = 40075016.686  # earth circumpherence
PROJECTION: str = "EPSG:3857"


def zoom_to_scale(zoom: ee.Number) -> ee.Number:
    zoom: ee.Number = ee.Number(zoom)
    return ee.Number(C).divide(ee.Number(2).pow(zoom)).divide(TILE_SIZE)

def scale_to_zoom(scale: ee.Number) -> ee.Number:
    scale: ee.Number = ee.Number(scale)
    zoom: ee.Number = ee.Number(C).divide(scale.multiply(TILE_SIZE)).log().divide(ee.Number(2).log())
    return zoom.ceil()

def to_radians(degrees: ee.Number) -> ee.Number:
    degrees: ee.Number = ee.Number(degrees)

    return(degrees.multiply(math.pi).divide(180))

def pixels_to_meters(px: ee.Number, py: ee.Number, zoom: ee.Number) -> ee.List:
    px: ee.Number = ee.Number(px)
    py: ee.Number = ee.Number(py)
    zoom = ee.Number(zoom)

    resolution: ee.Number = zoom_to_scale(zoom)
    x: ee.Number = px.multiply(resolution).subtract(ORIGIN)
    y: ee.Number = py.multiply(resolution).subtract(ORIGIN)
    return ee.List([x, y])

def meters_to_pixels(x: ee.Number, y: ee.Number, zoom: ee.Number) -> ee.List:
    x: ee.Number = ee.Number(x)
    y: ee.Number = ee.Number(y)
    zoom = ee.Number(zoom)

    resolution: ee.Number = zoom_to_scale(zoom)
    px: ee.Number = x.add(ORIGIN).divide(resolution)
    py: ee.Number = y.add(ORIGIN).divide(resolution)
    return ee.List([px, py])

def degrees_to_tiles(lon: ee.Number, lat: ee.Number, zoom: ee.Number) -> ee.List:
    lon: ee.Number = ee.Number(lon)
    lat: ee.Number = ee.Number(lat)
    zoom: ee.Number = ee.Number(zoom)

    tx: ee.Number = lon.add(180).divide(360).multiply(ee.Number(2).pow(zoom)).floor()
    ty: ee.Number = ee.Number(1).subtract(
        to_radians(lat).tan().add(
            ee.Number(1).divide(
                to_radians(lat).cos()
            )
        ).log().divide(ee.Number(math.pi))
    ).divide(2).multiply(ee.Number(2).pow(zoom)).floor()
    return ee.List([tx, ty])

def get_tile_bounds(tx: ee.Number, ty: ee.Number, zoom: ee.Number) -> ee.List:
    tx: ee.Number = ee.Number(tx)
    ty: ee.Number = ee.Number(ty)
    zoom: ee.Number = ee.Number(zoom)

    ty_flip: ee.Number = ee.Number(2).pow(zoom).subtract(ty).subtract(1) # TMS -> XYZ, flip y index
    min: ee.Number = pixels_to_meters(ee.Number(tx).multiply(TILE_SIZE), ty_flip.multiply(TILE_SIZE), zoom)
    max: ee.Number = pixels_to_meters(ee.Number(tx).add(1).multiply(TILE_SIZE), ty_flip.add(1).multiply(TILE_SIZE), zoom)
    return ee.List([min, max])

def get_tiles_for_geometry(geometry: ee.Geometry, zoom: ee.Number, opt_bounds: Optional[ee.Geometry] = None) -> ee.FeatureCollection:
    zoom: ee.Number = ee.Number(zoom)

    bounds: ee.Geometry = ee.Geometry(ee.Algorithms.If(opt_bounds, opt_bounds, geometry))
    bounds_list: ee.List = ee.List(bounds.bounds().coordinates().get(0))

    ll: ee.List = ee.List(bounds_list.get(0))
    ur: ee.List = ee.List(bounds_list.get(2))

    tmin: ee.List = ee.List(degrees_to_tiles(ee.Number(ll.get(0)), ee.Number(ll.get(1)), zoom))
    tmax: ee.List = ee.List(degrees_to_tiles(ee.Number(ur.get(0)), ee.Number(ur.get(1)), zoom))

    # create ranges for server-side mapping
    tx_range: ee.List = ee.List.sequence(tmin.get(0), ee.Number(tmax.get(0)).add(1))
    ty_range: ee.List = ee.List.sequence(tmax.get(1), ee.Number(tmin.get(1)).add(1))

    def create_tile(tx: ee.Number, ty: ee.Number, zoom: ee.Number) -> ee.Feature:
        """
        Add a tile to input tiles
        """
        tx: ee.Number = ee.Number(tx)
        ty: ee.Number = ee.Number(ty)
        zoom: ee.Number = ee.Number(zoom)

        tile_bounds: ee.List = get_tile_bounds(tx, ty, zoom)
        rect: ee.Geometry = ee.Geometry.Rectangle(tile_bounds, PROJECTION, False)
        return ee.Feature(rect).set({"tx": tx.format(), "ty": ty.format(), "zoom": zoom.format()})

    tiles: ee.List = ee.List(tx_range.map(lambda tx: ty_range.map(lambda ty: create_tile(tx=tx, ty=ty, zoom=zoom)))).flatten()

    return ee.FeatureCollection(tiles).filterBounds(geometry)
