#!/usr/bin/env python3
"""make_site_figure.py

Create a publication-quality “study site” figure from a downloaded satellite
preview image (JPG/PNG). The preview is treated as *non-georeferenced* unless a
matching GeoTIFF is provided.

Features
- Load a JPG/PNG preview image
- Optional crop (pixel bounds)
- Optional mild enhancement (contrast + sharpness) using Pillow
- Overlay a labeled WWTP site marker (pixel coordinates)
- Add north arrow
- Add scale bar ONLY if a matching GeoTIFF is provided and scale can be computed;
  otherwise add a visible note that scale bar is omitted
- Optional inset map of South Korea with a dot near Ulsan using Cartopy
  (fallback if Cartopy unavailable)
- Export high-resolution PNG and PDF
- Print a caption string with dataset citation (NASA HLSL30 v2.0 DOI)

Example
  /bin/python3 make_site_figure.py --input path/to/image.jpg --output ulsan_site_figure \
    --crop 120 60 680 430 --site-px 400 205 --site-label "Yongyeon WWTP" \
    --enhance --contrast 1.08 --sharpness 1.10 --inset --dpi 400

Notes
- Pixel coordinates are interpreted in the ORIGINAL image coordinate system
  (before cropping). If you crop, the marker position is automatically adjusted.
- The scale bar is only drawn when the provided GeoTIFF spatial scale can be
  reliably mapped to the preview image (dimension match check).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance


# -----------------------------
# Configuration / data classes
# -----------------------------


@dataclass(frozen=True)
class CropBox:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class SitePixel:
    x: int
    y: int


@dataclass(frozen=True)
class Args:
    input_path: Path
    output_base: Path
    crop: Optional[CropBox]
    site_px: SitePixel
    site_label: str
    enhance: bool
    contrast: float
    sharpness: float
    dpi: int
    inset: bool
    ulsan_lon: float
    ulsan_lat: float
    geotiff: Optional[Path]
    scalebar_km: float
    acquired_date: str
    label_dx: int
    label_dy: int


# -----------------------------
# Utilities
# -----------------------------


def _require_file(path: Path, what: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")


def _validate_crop(crop: CropBox, width: int, height: int) -> None:
    if not (0 <= crop.x0 < crop.x1 <= width):
        raise ValueError(f"Invalid crop x bounds: {crop} for width={width}")
    if not (0 <= crop.y0 < crop.y1 <= height):
        raise ValueError(f"Invalid crop y bounds: {crop} for height={height}")


def _validate_site_px(site: SitePixel, width: int, height: int) -> None:
    if not (0 <= site.x < width and 0 <= site.y < height):
        raise ValueError(f"site-px {site} outside image bounds {(width, height)}")


def load_image_rgb(path: Path) -> Image.Image:
    """Load an image as RGB (Pillow Image)."""
    _require_file(path, "Input image")
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def crop_image(img: Image.Image, crop: CropBox) -> Image.Image:
    """Crop an image with a pixel CropBox (left, upper, right, lower)."""
    return img.crop((crop.x0, crop.y0, crop.x1, crop.y1))


def enhance_image(img: Image.Image, contrast: float, sharpness: float) -> Image.Image:
    """Apply mild visual enhancement (contrast then sharpness).

    This is intended to improve readability without changing scientific meaning.
    Keep factors close to 1.0.
    """
    img2 = ImageEnhance.Contrast(img).enhance(contrast)
    img3 = ImageEnhance.Sharpness(img2).enhance(sharpness)
    return img3


def adjust_site_for_crop(site: SitePixel, crop: Optional[CropBox]) -> SitePixel:
    """Convert site pixel coordinates from original-image space to cropped-image space."""
    if crop is None:
        return site
    return SitePixel(x=site.x - crop.x0, y=site.y - crop.y0)


# -----------------------------
# Optional scale bar (GeoTIFF)
# -----------------------------


def compute_meters_per_pixel_from_geotiff(geotiff: Path) -> Optional[float]:
    """Compute meters-per-pixel from a GeoTIFF.

    Returns
    - meters_per_pixel if reliably computable
    - None otherwise

    Notes
    - If the GeoTIFF CRS is projected (meters), we use affine transform.
    - If the CRS is geographic (degrees), we attempt a geodesic conversion using
      pyproj (typically installed with rasterio). If unavailable, return None.
    """
    try:
        import rasterio  # type: ignore
    except Exception:
        return None

    _require_file(geotiff, "GeoTIFF")

    with rasterio.open(geotiff) as ds:
        transform = ds.transform
        crs = ds.crs
        # Pixel sizes in CRS units
        px_x = float(abs(transform.a))
        px_y = float(abs(transform.e))
        px = float((px_x + px_y) / 2.0)

        if crs is None:
            return None

        # Projected CRS: assume units are meters.
        if getattr(crs, "is_projected", False):
            return px

        # Geographic CRS (degrees): approximate meters using geodesic.
        if getattr(crs, "is_geographic", False):
            try:
                from pyproj import Geod  # type: ignore
            except Exception:
                return None

            # Use dataset center for conversion.
            cx = ds.width / 2.0
            cy = ds.height / 2.0
            lon0, lat0 = transform * (cx, cy)
            geod = Geod(ellps="WGS84")
            # 1 pixel in x-direction (degrees)
            lon1 = lon0 + px_x
            _, _, dist_x = geod.inv(lon0, lat0, lon1, lat0)
            # 1 pixel in y-direction (degrees)
            lat1 = lat0 + px_y
            _, _, dist_y = geod.inv(lon0, lat0, lon0, lat1)
            return float((abs(dist_x) + abs(dist_y)) / 2.0)

    return None


def geotiff_dimensions(geotiff: Path) -> Optional[Tuple[int, int]]:
    try:
        import rasterio  # type: ignore
    except Exception:
        return None

    with rasterio.open(geotiff) as ds:
        return int(ds.width), int(ds.height)


# -----------------------------
# Plotting helpers
# -----------------------------


def set_paper_style() -> None:
    """Set conservative, paper-friendly Matplotlib defaults."""
    mpl.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_north_arrow(ax: plt.Axes, loc: Tuple[float, float] = (0.94, 0.12)) -> None:
    """Draw a simple north arrow in axes-fraction coordinates."""
    x, y = loc
    ax.annotate(
        "N",
        xy=(x, y + 0.08),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="black", linewidth=0.8),
    )
    ax.annotate(
        "",
        xy=(x, y + 0.07),
        xytext=(x, y - 0.02),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.2, shrinkA=0, shrinkB=0),
    )


def _auto_label_offset(site: SitePixel, w: int, h: int) -> Tuple[int, int, str, str]:
    """Choose a clean label offset based on site position in image bounds."""
    dx = 22 if site.x < 0.70 * w else -22
    dy = -22 if site.y > 0.30 * h else 22
    ha = "left" if dx > 0 else "right"
    va = "top" if dy < 0 else "bottom"
    return dx, dy, ha, va


def draw_site_marker(
    ax: plt.Axes,
    site: SitePixel,
    label: str,
    image_size: Tuple[int, int],
    label_dx: Optional[int] = None,
    label_dy: Optional[int] = None,
) -> None:
    """Draw a site marker and label box."""
    ax.scatter(
        [site.x],
        [site.y],
        s=55,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
    )
    ax.scatter(
        [site.x],
        [site.y],
        s=10,
        marker="o",
        facecolor="black",
        edgecolor="black",
        zorder=6,
    )

    # Label offset with a leader line (auto if not explicitly provided)
    w, h = image_size
    auto_dx, auto_dy, auto_ha, auto_va = _auto_label_offset(site, w, h)
    dx = int(auto_dx if label_dx is None else label_dx)
    dy = int(auto_dy if label_dy is None else label_dy)
    ha = auto_ha if label_dx is None else ("left" if dx >= 0 else "right")
    va = auto_va if label_dy is None else ("bottom" if dy >= 0 else "top")

    ax.annotate(
        label,
        xy=(site.x, site.y),
        xytext=(site.x + dx, site.y + dy),
        textcoords="data",
        ha=ha,
        va=va,
        fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.8),
        arrowprops=dict(arrowstyle="-", color="black", linewidth=0.8),
        zorder=7,
    )


def draw_scale_bar(
    ax: plt.Axes,
    meters_per_pixel: float,
    length_km: float,
    loc: Tuple[float, float] = (0.06, 0.08),
) -> None:
    """Draw a scale bar in image pixel coordinates.

    Parameters
    - meters_per_pixel: meters per pixel
    - length_km: desired scale bar length in kilometers
    - loc: (x_frac, y_frac) location in axes fraction
    """
    length_m = float(length_km * 1000.0)
    length_px = length_m / float(meters_per_pixel)

    # Convert loc (axes fraction) to data coords
    x0_frac, y0_frac = loc
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + x0_frac * (xlim[1] - xlim[0])
    y0 = ylim[0] + y0_frac * (ylim[1] - ylim[0])

    x1 = x0 + length_px

    ax.plot([x0, x1], [y0, y0], color="black", linewidth=2.2, solid_capstyle="butt", zorder=8)
    ax.plot([x0, x0], [y0 - 6, y0 + 6], color="black", linewidth=1.2, zorder=8)
    ax.plot([x1, x1], [y0 - 6, y0 + 6], color="black", linewidth=1.2, zorder=8)

    ax.text(
        (x0 + x1) / 2.0,
        y0 - 12,
        f"{length_km:g} km",
        ha="center",
        va="top",
        fontsize=9,
        color="black",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        zorder=9,
    )


def draw_scale_note(ax: plt.Axes, note: str, loc: Tuple[float, float] = (0.06, 0.08)) -> None:
    ax.text(
        loc[0],
        loc[1],
        note,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.8,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=0.6),
        zorder=10,
    )


def add_inset_map(
    fig: plt.Figure,
    ulsan_lon: float,
    ulsan_lat: float,
    enabled: bool,
) -> None:
    """Add an inset map of South Korea with a dot near Ulsan.

    Uses Cartopy if available; otherwise shows a fallback box with instructions.
    """
    if not enabled:
        return

    # Inset position in figure fraction
    inset_rect = [0.03, 0.64, 0.28, 0.33]  # left, bottom, width, height

    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore

        ax_in = fig.add_axes(inset_rect, projection=ccrs.PlateCarree())
        ax_in.set_extent([124.0, 132.5, 33.0, 39.5], crs=ccrs.PlateCarree())

        ax_in.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f4f6f8")
        ax_in.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e9ecef")
        ax_in.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
        ax_in.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)

        ax_in.plot(
            [ulsan_lon],
            [ulsan_lat],
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markerfacecolor="white",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        ax_in.text(
            ulsan_lon + 0.25,
            ulsan_lat + 0.10,
            "Ulsan",
            transform=ccrs.PlateCarree(),
            fontsize=8.5,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="black", linewidth=0.5),
            zorder=6,
        )

        ax_in.set_title("South Korea", fontsize=9, pad=2)
        ax_in.outline_patch.set_linewidth(0.8)

    except Exception:
        ax_in = fig.add_axes(inset_rect)
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_frame_on(True)
        ax_in.text(
            0.5,
            0.55,
            "Inset map unavailable\nInstall cartopy",
            ha="center",
            va="center",
            transform=ax_in.transAxes,
            fontsize=8.5,
        )
        ax_in.text(
            0.5,
            0.28,
            f"Ulsan approx\n({ulsan_lat:.3f}°N, {ulsan_lon:.3f}°E)",
            ha="center",
            va="center",
            transform=ax_in.transAxes,
            fontsize=7.5,
        )


# -----------------------------
# Caption
# -----------------------------


def build_caption(site_label: str, acquired_date: str, access_date: str = "YYYY-MM-DD") -> str:
    """Return an ACS-style caption string with dataset citation."""
    return (
        f"Study site ({site_label}, Ulsan, South Korea) shown on a true-color composite derived from "
        f"NASA Harmonized Landsat and Sentinel-2 (HLS) surface reflectance product "
        f"(HLSL30 v2.0; 30 m), acquired {acquired_date}. Image cropped/annotated for clarity. "
        f"Source: NASA EOSDIS Land Processes DAAC; DOI: 10.5067/HLS/HLSL30.002; accessed {access_date}."
    )


# -----------------------------
# Main
# -----------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> Args:
    p = argparse.ArgumentParser(description="Create a publication-quality study site figure from a satellite preview image.")

    p.add_argument("--input", required=True, type=Path, help="Path to input JPG/PNG preview image")
    p.add_argument("--output", required=True, type=Path, help="Output basename (writes .png and .pdf)")

    p.add_argument("--crop", nargs=4, type=int, metavar=("x0", "y0", "x1", "y1"), help="Optional crop box in pixels")

    p.add_argument("--site-px", nargs=2, required=True, type=int, metavar=("x", "y"), help="WWTP site pixel coords (in original image coords)")
    p.add_argument("--site-label", required=True, type=str, help="Site label text")

    p.add_argument("--enhance", action="store_true", help="Apply mild contrast+sharpness enhancements")
    p.add_argument("--contrast", type=float, default=1.08, help="Contrast factor (default 1.08)")
    p.add_argument("--sharpness", type=float, default=1.10, help="Sharpness factor (default 1.10)")

    p.add_argument("--dpi", type=int, default=300, help="Export DPI (default 300)")

    p.add_argument("--inset", action="store_true", help="Add inset map (Cartopy preferred)")
    p.add_argument("--ulsan-lon", type=float, default=129.311, help="Ulsan longitude for inset dot")
    p.add_argument("--ulsan-lat", type=float, default=35.539, help="Ulsan latitude for inset dot")

    p.add_argument("--geotiff", type=Path, default=None, help="Optional GeoTIFF for scale bar computation")
    p.add_argument("--scalebar-km", type=float, default=2.0, help="Scale bar length (km) if GeoTIFF usable")

    p.add_argument("--acquired-date", type=str, default="May 1, 2023", help="Acquisition date string for caption")
    p.add_argument("--label-dx", type=int, default=None, help="Optional manual label x-offset in pixels")
    p.add_argument("--label-dy", type=int, default=None, help="Optional manual label y-offset in pixels")

    ns = p.parse_args(argv)

    crop = CropBox(*ns.crop) if ns.crop is not None else None
    return Args(
        input_path=ns.input,
        output_base=ns.output,
        crop=crop,
        site_px=SitePixel(*ns.site_px),
        site_label=str(ns.site_label),
        enhance=bool(ns.enhance),
        contrast=float(ns.contrast),
        sharpness=float(ns.sharpness),
        dpi=int(ns.dpi),
        inset=bool(ns.inset),
        ulsan_lon=float(ns.ulsan_lon),
        ulsan_lat=float(ns.ulsan_lat),
        geotiff=ns.geotiff,
        scalebar_km=float(ns.scalebar_km),
        acquired_date=str(ns.acquired_date),
        label_dx=ns.label_dx,
        label_dy=ns.label_dy,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    set_paper_style()

    img = load_image_rgb(args.input_path)
    width, height = img.size

    # Validate crop and site pixel in original image
    if args.crop is not None:
        _validate_crop(args.crop, width, height)
    _validate_site_px(args.site_px, width, height)

    # Apply crop
    img_for_plot = crop_image(img, args.crop) if args.crop is not None else img

    # Enhance (optional)
    if args.enhance:
        img_for_plot = enhance_image(img_for_plot, args.contrast, args.sharpness)

    # Adjust site pixel for crop
    site_plot = adjust_site_for_crop(args.site_px, args.crop)

    # Validate that adjusted site lies within final image
    w2, h2 = img_for_plot.size
    _validate_site_px(site_plot, w2, h2)

    # Create figure (size chosen to be clean in ACS layouts)
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    ax.imshow(np.asarray(img_for_plot))
    ax.set_axis_off()

    draw_site_marker(
        ax,
        site_plot,
        args.site_label,
        image_size=(w2, h2),
        label_dx=args.label_dx,
        label_dy=args.label_dy,
    )
    draw_north_arrow(ax)

    # Scale bar
    drew_scalebar = False
    if args.geotiff is not None:
        _require_file(args.geotiff, "GeoTIFF")
        dims = geotiff_dimensions(args.geotiff)
        if dims is None:
            draw_scale_note(ax, "scale bar omitted (install rasterio)")
        else:
            # Only draw if dimensions match either original or final image
            ok = dims == (width, height) or dims == (w2, h2)
            mpp = compute_meters_per_pixel_from_geotiff(args.geotiff)
            if ok and (mpp is not None) and np.isfinite(mpp) and mpp > 0:
                draw_scale_bar(ax, mpp, args.scalebar_km)
                drew_scalebar = True
            else:
                draw_scale_note(ax, "scale bar omitted (GeoTIFF does not match preview)")
    if args.geotiff is None and not drew_scalebar:
        draw_scale_note(ax, "scale bar omitted (non-georeferenced preview)")

    # Inset map
    add_inset_map(fig, args.ulsan_lon, args.ulsan_lat, enabled=args.inset)

    # Export
    out_png = args.output_base.with_suffix(".png")
    out_pdf = args.output_base.with_suffix(".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Caption
    caption = build_caption(args.site_label, args.acquired_date)
    print(caption)


if __name__ == "__main__":
    main()
