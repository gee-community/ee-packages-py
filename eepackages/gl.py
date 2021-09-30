import ee

def smoothStep(edge0: float, edge1: float, x: ee.Image):
    t: ee.Image = x.subtract(edge0).divide(ee.Image(edge1).subtract(edge0)).clamp(0, 1)
    return t.multiply(t).multiply(ee.Image.constant(3).subtract(ee.Image.constant(2).multiply(t)))
