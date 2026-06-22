"""STAC catalog and collection configuration for PlaneSight data sources.

The default catalog is AWS Earth Search (Element84), which is genuinely anonymous
- no token signing required (decision D13). Microsoft Planetary Computer would
need SAS signing and is intentionally not the default.

Collection ids and asset keys below were verified against the live Earth Search
v1 API (June 2026).
"""

from __future__ import annotations

#: AWS Earth Search v1 STAC API (anonymous).
EARTH_SEARCH_V1 = "https://earth-search.aws.element84.com/v1"

# --- Collections (verified present on Earth Search v1) ---
SENTINEL2_L2A = "sentinel-2-l2a"
COP_DEM_GLO_30 = "cop-dem-glo-30"

#: Copernicus GLO-30 elevation COG asset key.
COP_DEM_ASSET = "data"

#: Sentinel-2 L2A COG band asset keys (the non-"-jp2" variants are COGs).
S2_BANDS = {
    "coastal": "coastal",  # B1
    "blue": "blue",  # B2
    "green": "green",  # B3
    "red": "red",  # B4
    "rededge1": "rededge1",  # B5
    "rededge2": "rededge2",  # B6
    "rededge3": "rededge3",  # B7
    "nir": "nir",  # B8
    "nir08": "nir08",  # B8A
    "nir09": "nir09",  # B9
    "swir16": "swir16",  # B11
    "swir22": "swir22",  # B12
    "scl": "scl",  # Scene Classification (cloud mask) - used by P0d
}

#: STAC property carrying scene cloud cover (supports the query-extension filter).
CLOUD_COVER_PROP = "eo:cloud_cover"

#: Default maximum Sentinel-2 scene cloud cover (percent). We strongly prefer
#: near-clear imagery; geology reads best with minimal cloud AND in the dry
#: season (less vegetation/snow).
DEFAULT_MAX_CLOUD = 5.0

#: Upper bound when escalating the cloud threshold for cloudy AOIs that have no
#: scene under DEFAULT_MAX_CLOUD in the requested window.
CLOUD_CEILING = 20.0
