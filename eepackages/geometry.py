import ee
import math

def createTransectAtCentroid(line, length, crs, maxError):
    '''
    Creates transect in the middle of line geometry
    '''
    if not crs:
        crs = 'EPSG:3857'
  
    if not maxError:
        maxError = 1
  
    line = ee.Geometry(line).transform(crs, maxError)
    origin = line.centroid(maxError, crs)
  
    length = ee.Number(length)

    # compute angle from two points
    coords = line.coordinates()
    pt0 = coords.slice(0, 1)
    pt1 = coords.slice(-1)
    delta = ee.Array(pt1).subtract(pt0)
    dx = delta.get([0, 0])
    dy = delta.get([0, 1])
    angle = dx.atan2(dy).add(math.pi / 2)
  
    ptOrigin = ee.Array([origin.coordinates()]).transpose()
  
    # get coordinates as a list
    proj1 = origin.projection().translate(length.multiply(-0.5), 0)
    pt1 = ee.Array([origin.transform(proj1).coordinates()]).transpose()

    # translate
    proj2 = origin.projection().translate(length.multiply(0.5), 0)
    pt2 = ee.Array([origin.transform(proj2).coordinates()]).transpose()

    # define rotation matrix
    cosa = angle.cos()
    sina = angle.sin()
    M = ee.Array([
    [cosa, sina.multiply(-1)], 
    [sina, cosa]
    ])

    # rotate
    pt1 = M.matrixMultiply(pt1.subtract(ptOrigin)).add(ptOrigin)
    pt2 = M.matrixMultiply(pt2.subtract(ptOrigin)).add(ptOrigin)

    # get points
    pt1 = pt1.transpose().project([1]).toList()
    pt2 = pt2.transpose().project([1]).toList()

    # construct line
    line = ee.Algorithms.GeometryConstructors.LineString([pt1, pt2], ee.Projection(crs))

    return line

def createVector(origin, angle, length):
  # get coordinates as a list
  pt1 = ee.Array([ee.Feature(origin).geometry().coordinates()]).transpose()

  # translate
  proj = origin.projection().translate(length, 0)
  pt2 = ee.Array([origin.transform(proj).coordinates()]).transpose()
  
  # define rotation matrix
  angle = ee.Number(angle).multiply(math.pi).divide(180)
  cosa = angle.cos()
  sina = angle.sin()
  M = ee.Array([
    [cosa, sina.multiply(-1)], 
    [sina, cosa]
  ])

  # rotate
  pt2 = M.matrixMultiply(pt2.subtract(pt1)).add(pt1)

  # get end point  
  pt2 = pt2.transpose().project([1]).toList()
  
  # construct line
  line = ee.Algorithms.GeometryConstructors.LineString([origin, pt2], origin.projection())  

  return line

def angle(pt0, pt1):
    pt0 = ee.List(pt0)
    pt1 = ee.List(pt1)
  
    x0 = ee.Number(pt0.get(0))
    x1 = ee.Number(pt1.get(0))
    y0 = ee.Number(pt0.get(1))
    y1 = ee.Number(pt1.get(1))

    dy = y1.subtract(y0)
    dx = x1.subtract(x0)
  
    return dy.atan2(dx)

