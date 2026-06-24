"""DEM/imagery derivative engine - the multi-band detection input stack.

Computes co-registered DEM-derived (terrain) and Sentinel-2-derived (spectral)
bands using numpy/scipy only (decision D11), and assembles them into the stack
the trace detector consumes. Implemented in Phase 0 issue planesight-6s7; see
ARCHITECTURE.md S7.2. Co-registration/resampling lives at the GDAL I/O edge, not
here, so every derivative is CI-testable on plain arrays.
"""

from .stack import (
    DEFAULT_SPECTRAL,
    DEFAULT_TERRAIN,
    SPECTRAL_BANDS,
    TERRAIN_BANDS,
    assemble,
    build_spectral_stack,
    build_terrain_stack,
    normalize01,
)

__all__ = [
    "TERRAIN_BANDS",
    "SPECTRAL_BANDS",
    "DEFAULT_TERRAIN",
    "DEFAULT_SPECTRAL",
    "build_terrain_stack",
    "build_spectral_stack",
    "assemble",
    "normalize01",
]
