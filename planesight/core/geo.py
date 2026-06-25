"""Geographic helpers: per-AOI metric CRS selection (the gj9 v1 policy).

Plane-fit geometry must be computed in a METRIC CRS, so every AOI is reprojected to
a local UTM zone. v1 picks the zone containing the AOI centroid; the fuller policy
(custom transverse-Mercator / Lambert, or tiling very wide AOIs that straddle zones)
is tracked as planesight-gj9.
"""

from __future__ import annotations


def utm_epsg(bbox) -> int:
    """EPSG code of the UTM zone at the AOI centroid.

    Args:
        bbox: ``[west, south, east, north]`` in lon/lat degrees (EPSG:4326).

    Returns:
        The EPSG code: ``326xx`` (north) or ``327xx`` (south), ``xx`` = UTM zone.
    """
    west, south, east, north = bbox
    lon = 0.5 * (west + east)
    lat = 0.5 * (south + north)
    zone = int((lon + 180.0) // 6) + 1
    zone = min(max(zone, 1), 60)            # clamp at the antimeridian
    return (32600 if lat >= 0 else 32700) + zone
