"""Capture-rig geometry for the synthetic renderer.

Ported from the research project's light_geometry.py (synthetic fixed-ring
section, R22). The real-rig calibration JSONs / /Storage paths aren't available
here, so this reimplements only the formula-driven synthetic ring:

    horizontal ring radius is the invariant (RING_RADIUS_MM);
    slant distance D and ring height z are DERIVED from elevation:
        D = RING_RADIUS_MM / cos(elev)
        z = RING_RADIUS_MM * tan(elev)

Our material dataset uses ELEVATION_DEG = 60 (steeper / more top-down than the
height project's 30deg ring), 16 azimuths, R/G/B per azimuth.

Positions are in millimetres in the same world frame as the gerber2blend board
mesh (board centred near origin, +z up), so they drop straight into Blender.
"""

from __future__ import annotations

import math

RING_RADIUS_MM = 385.0     # measured real-rig radius (8.75in)
ELEVATION_DEG = 60.0       # material-dataset ring elevation (from board plane)
N_LIGHTS = 16              # azimuth positions
AZIMUTH_STEP_DEG = 360.0 / N_LIGHTS   # 22.5 deg

# LED colours captured per azimuth. Pure primaries (linear RGB); the mono
# sensor integrates the colour-lit render to one intensity channel.
LED_COLORS: dict[str, tuple[float, float, float]] = {
    "R": (1.0, 0.0, 0.0),
    "G": (0.0, 1.0, 0.0),
    "B": (0.0, 0.0, 1.0),
}


def slant_distance_mm(elev_deg: float = ELEVATION_DEG,
                      ring_radius_mm: float = RING_RADIUS_MM) -> float:
    """Light-to-board-centre distance: D = R / cos(elev)."""
    return ring_radius_mm / math.cos(math.radians(elev_deg))


def ring_height_mm(elev_deg: float = ELEVATION_DEG,
                   ring_radius_mm: float = RING_RADIUS_MM) -> float:
    """Height of the ring above the board plane: z = R * tan(elev)."""
    return ring_radius_mm * math.tan(math.radians(elev_deg))


def light_positions(elev_deg: float = ELEVATION_DEG,
                    n_lights: int = N_LIGHTS,
                    az_offset_deg: float = 0.0,
                    ring_radius_mm: float = RING_RADIUS_MM
                    ) -> list[tuple[float, float, float]]:
    """Fixed-ring light positions in mm, CCW from +x (synthetic convention).

    Horizontal radius = ring_radius_mm; z derived from elevation.
    Returns a list of (x, y, z) tuples, one per azimuth.
    """
    D = slant_distance_mm(elev_deg, ring_radius_mm)
    rad_el = math.radians(elev_deg)
    z = D * math.sin(rad_el)
    horiz = D * math.cos(rad_el)          # == ring_radius_mm
    out = []
    for i in range(n_lights):
        az = math.radians(az_offset_deg + i * (360.0 / n_lights))
        out.append((horiz * math.cos(az), horiz * math.sin(az), z))
    return out


def relative_intensity(elev_deg: float = ELEVATION_DEG,
                       ring_radius_mm: float = RING_RADIUS_MM) -> float:
    """D**2 scale that keeps board-centre irradiance ~constant across
    elevations (the synthetic analogue of per-capture auto-exposure)."""
    return slant_distance_mm(elev_deg, ring_radius_mm) ** 2


if __name__ == "__main__":
    print(f"elevation={ELEVATION_DEG} radius={RING_RADIUS_MM}mm")
    print(f"slant D={slant_distance_mm():.1f}mm  ring height z={ring_height_mm():.1f}mm")
    for i, p in enumerate(light_positions()):
        print(f"  L{i:02d} az={i*AZIMUTH_STEP_DEG:5.1f}  "
              f"({p[0]:7.1f},{p[1]:7.1f},{p[2]:7.1f}) mm")
