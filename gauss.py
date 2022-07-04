# coding: utf-8
import numpy as np


def gaussianRadius(height, width, min_overlap=0.7):
    """ Get min gauss radius in three situations """
    h, w = float(height), float(width)

    a1 = 1.0
    b1 = (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2.0

    a2 = 4.0
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2.0

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2.0

    return min(r1, r2, r3)


def gaussDistribution2D(shape, sigma=1, eps=None):
    """ Return standard normal distribution matrix """
    m, n = [(s - 1.) / 2. for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    if eps:
        h[h < eps] = 0
    else:
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def createGaussHeatmap(heatmap, center, radius, scale=1.0):
    """ Mask heatmap use gauss distribution around center """
    diameter = 2 * radius + 1
    gaussian = gaussDistribution2D((diameter, diameter), sigma=diameter / 6)

    h, w = heatmap.shape[0], heatmap.shape[1]
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    left = min(cx, r)
    right = min(w - cx, r + 1)
    top = min(cy, r)
    bottom = min(h - cy, r + 1)
    masked_heatmap = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    masked_gaussian = gaussian[r - top:r + bottom, r - left:r + right]

    assert min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0
    np.maximum(masked_heatmap, masked_gaussian * scale, out=masked_heatmap)

    return heatmap

