#!/usr/bin/env python3
"""Create a clean, text-free study map with marker geometry only.

This script does NOT use the original satellite image pixels. It builds a new map
from cartographic layers (Cartopy Natural Earth) and draws only symbols (points,
rings, and optional study-area box) so labels can be added later in Photoshop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw_fallback_base(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    """Draw a stylized lon/lat base map without cartopy (no labels/text)."""
    xmin, xmax, ymin, ymax = extent

    # Ocean background
    ax.set_facecolor("#e8f1fb")

    # Rough polygon sketch of SE Korean coast/land in this local extent
    land_poly = [
        (128.92, 35.92), (129.00, 35.90), (129.08, 35.88), (129.15, 35.84),
        (129.22, 35.80), (129.28, 35.74), (129.34, 35.66), (129.39, 35.57),
        (129.42, 35.49), (129.43, 35.40), (129.40, 35.31), (129.35, 35.24),
        (129.28, 35.18), (129.20, 35.14), (129.11, 35.12), (129.04, 35.13),
        (128.99, 35.17), (128.96, 35.24), (128.94, 35.33), (128.93, 35.43),
        (128.92, 35.54), (128.91, 35.66), (128.90, 35.77), (128.91, 35.86),
        (128.92, 35.92),
    ]
    xs = [p[0] for p in land_poly]
    ys = [p[1] for p in land_poly]
    ax.fill(xs, ys, facecolor="#edf2e6", edgecolor="#425466", linewidth=1.0, zorder=1)

    # Approximate inlets/coastal indentations near Ulsan
    inlet1 = [(129.16, 35.60), (129.20, 35.59), (129.22, 35.56), (129.19, 35.54), (129.15, 35.55), (129.14, 35.58)]
    inlet2 = [(129.28, 35.55), (129.31, 35.54), (129.33, 35.50), (129.30, 35.48), (129.27, 35.50)]
    for inlet in (inlet1, inlet2):
        ixs = [p[0] for p in inlet]
        iys = [p[1] for p in inlet]
        ax.fill(ixs, iys, facecolor="#e8f1fb", edgecolor="#6f8298", linewidth=0.8, zorder=2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")


def build_map(
    output_base: Path,
    dpi: int,
    center_lon: float,
    center_lat: float,
    extent_deg: float,
    wwtp_lon: float,
    wwtp_lat: float,
    ulsan_lon: float,
    ulsan_lat: float,
    draw_box: bool,
) -> None:
    set_style()
    fig = plt.figure(figsize=(8.2, 6.8))
    extent = (
        center_lon - extent_deg,
        center_lon + extent_deg,
        center_lat - extent_deg,
        center_lat + extent_deg,
    )

    use_cartopy = True
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
    except Exception:
        use_cartopy = False

    if use_cartopy:
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection=ccrs.PlateCarree())
        ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e8f1fb", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#edf2e6", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8, edgecolor="#425466", zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, edgecolor="#607085", zorder=2)
    else:
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        _draw_fallback_base(ax, extent)

    # Optional area box around study vicinity (unlabeled)
    if draw_box:
        box_half = 0.06
        xs = [wwtp_lon - box_half, wwtp_lon + box_half, wwtp_lon + box_half, wwtp_lon - box_half, wwtp_lon - box_half]
        ys = [wwtp_lat - box_half, wwtp_lat - box_half, wwtp_lat + box_half, wwtp_lat + box_half, wwtp_lat - box_half]
        if use_cartopy:
            ax.plot(xs, ys, color="#1d3557", linewidth=1.4, transform=ccrs.PlateCarree(), zorder=4)
        else:
            ax.plot(xs, ys, color="#1d3557", linewidth=1.4, zorder=4)

    # Marker 1: WWTP (primary)
    ax.scatter(
        [wwtp_lon],
        [wwtp_lat],
        s=115,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        transform=ccrs.PlateCarree() if use_cartopy else None,
        zorder=7,
    )
    ax.scatter(
        [wwtp_lon],
        [wwtp_lat],
        s=36,
        facecolor="#d90429",
        edgecolor="black",
        linewidth=0.6,
        transform=ccrs.PlateCarree() if use_cartopy else None,
        zorder=8,
    )

    # Concentric rings around WWTP to indicate area-of-interest (no text)
    for size, lw, alpha in [(380, 1.4, 0.95), (760, 1.2, 0.7), (1250, 1.0, 0.45)]:
        ax.scatter(
            [wwtp_lon],
            [wwtp_lat],
            s=size,
            facecolors="none",
            edgecolors="#d90429",
            linewidths=lw,
            alpha=alpha,
            transform=ccrs.PlateCarree() if use_cartopy else None,
            zorder=6,
        )

    # Marker 2: Ulsan urban reference (secondary, unlabeled)
    ax.scatter(
        [ulsan_lon],
        [ulsan_lat],
        s=55,
        marker="D",
        facecolor="#1d3557",
        edgecolor="white",
        linewidth=0.7,
        transform=ccrs.PlateCarree() if use_cartopy else None,
        zorder=7,
    )

    # Neatline only; no ticks/grid/text
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#2f3e4d")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a clean no-text study map with markers only.")
    p.add_argument("--output", type=Path, required=True, help="Output basename (without extension)")
    p.add_argument("--dpi", type=int, default=600, help="Export DPI")

    p.add_argument("--center-lon", type=float, default=129.20)
    p.add_argument("--center-lat", type=float, default=35.56)
    p.add_argument("--extent-deg", type=float, default=0.36, help="Half-width/height in degrees")

    p.add_argument("--wwtp-lon", type=float, default=129.1269)
    p.add_argument("--wwtp-lat", type=float, default=35.56233)
    p.add_argument("--ulsan-lon", type=float, default=129.311)
    p.add_argument("--ulsan-lat", type=float, default=35.539)

    p.add_argument("--no-box", action="store_true", help="Disable study-area box")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_map(
        output_base=args.output,
        dpi=int(args.dpi),
        center_lon=float(args.center_lon),
        center_lat=float(args.center_lat),
        extent_deg=float(args.extent_deg),
        wwtp_lon=float(args.wwtp_lon),
        wwtp_lat=float(args.wwtp_lat),
        ulsan_lon=float(args.ulsan_lon),
        ulsan_lat=float(args.ulsan_lat),
        draw_box=not bool(args.no_box),
    )


if __name__ == "__main__":
    main()
