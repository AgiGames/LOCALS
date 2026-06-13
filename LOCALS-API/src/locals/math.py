import numpy as np

import math

def smooth_curve(points, factor=0.9):
    # applies exponential smoothing to a curve
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

def closest_factors(n):
    d = math.isqrt(n)  # integer sqrt
    while n % d != 0:
        d -= 1
    return d, n // d

# function that calculates the pearson correlation coefficient given a list of points
def pearson_corr(points):
    points_array = np.array(points)

    x = points_array[:, 0]
    y = points_array[:, 1]

    corr = np.corrcoef(x, y)[0, 1]
    return corr