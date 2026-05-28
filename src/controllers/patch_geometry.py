"""Patch geometry helpers for cavity-specific local control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchClassification:
    """Geometry-only patch labels used by cavity-aware control."""

    patch_type: str
    near_wall: bool
    centerline_band: bool
    tags: tuple[str, ...]


def classify_patch(
    bounds: tuple[float, float, float, float, float | None, float | None],
    domain_bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    near_wall_width: float = 0.10,
    centerline_width: float = 0.06,
    boundary_margin: float = 1.0e-9,
) -> PatchClassification:
    """Classify a patch using only its bounds and domain geometry."""
    x0, x1, y0, y1 = (float(bounds[i]) for i in range(4))
    dx0, dx1, dy0, dy1 = (float(v) for v in domain_bounds)
    width = max(dx1 - dx0, 1.0e-12)
    height = max(dy1 - dy0, 1.0e-12)
    wall_width = max(float(near_wall_width), 0.0) * min(width, height)
    band_width = max(float(centerline_width), 0.0) * min(width, height)

    touches_left = x0 <= dx0 + float(boundary_margin)
    touches_right = x1 >= dx1 - float(boundary_margin)
    touches_bottom = y0 <= dy0 + float(boundary_margin)
    touches_top = y1 >= dy1 - float(boundary_margin)
    touches_side = touches_left or touches_right
    touches_wall = touches_side or touches_bottom or touches_top
    touches_corner = touches_side and (touches_bottom or touches_top)

    near_left = x0 <= dx0 + wall_width
    near_right = x1 >= dx1 - wall_width
    near_bottom = y0 <= dy0 + wall_width
    near_top = y1 >= dy1 - wall_width
    near_wall = near_left or near_right or near_bottom or near_top

    x_mid = 0.5 * (dx0 + dx1)
    y_mid = 0.5 * (dy0 + dy1)
    overlaps_x_mid = x0 <= x_mid + band_width and x1 >= x_mid - band_width
    overlaps_y_mid = y0 <= y_mid + band_width and y1 >= y_mid - band_width
    centerline_band = overlaps_x_mid or overlaps_y_mid

    if touches_corner:
        patch_type = "corner"
    elif touches_top:
        patch_type = "lid"
    elif touches_bottom:
        patch_type = "bottom_wall"
    elif touches_side:
        patch_type = "sidewall"
    elif centerline_band:
        patch_type = "centerline_band"
    else:
        patch_type = "interior"

    tags = [patch_type]
    if near_wall and "near_wall" not in tags:
        tags.append("near_wall")
    if centerline_band and "centerline_band" not in tags:
        tags.append("centerline_band")
    return PatchClassification(patch_type=patch_type, near_wall=near_wall, centerline_band=centerline_band, tags=tuple(tags))
