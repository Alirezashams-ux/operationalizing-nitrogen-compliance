#!/usr/bin/env python3
"""Create a publication-quality 'Site Description and Dataset' overview figure.

This script builds a completely new composite figure using the provided satellite
image as the visual source and adds structured annotation panels suitable for
manuscript Section 2.1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageEnhance


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_card(ax: plt.Axes, title: str, body: str, title_bg: str = "#0f2f57") -> None:
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor="#f7f9fc",
            edgecolor="#c8d2e1",
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, 0.83),
            1.0,
            0.17,
            transform=ax.transAxes,
            facecolor=title_bg,
            edgecolor=title_bg,
            linewidth=0.0,
        )
    )
    ax.text(
        0.03,
        0.915,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.79,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#1b2430",
        fontsize=9.2,
        linespacing=1.28,
        wrap=True,
    )


def draw_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N",
        xy=(0.94, 0.13),
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
        xy=(0.94, 0.12),
        xytext=(0.94, 0.03),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.2),
    )


def draw_marker(ax: plt.Axes, x: int, y: int, label: str, w: int, h: int) -> None:
    ax.scatter([x], [y], s=70, facecolor="white", edgecolor="black", linewidth=1.2, zorder=6)
    ax.scatter([x], [y], s=16, facecolor="#d7263d", edgecolor="black", linewidth=0.5, zorder=7)

    dx = 24 if x < 0.68 * w else -24
    dy = -22 if y > 0.28 * h else 22
    ha = "left" if dx > 0 else "right"
    va = "top" if dy < 0 else "bottom"

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        ha=ha,
        va=va,
        fontsize=9.5,
        color="black",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.8),
        arrowprops=dict(arrowstyle="-", color="black", linewidth=0.8),
        zorder=8,
    )


def add_inset_map(fig: plt.Figure, lon: float, lat: float) -> None:
    inset_rect = [0.06, 0.64, 0.22, 0.24]
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore

        ax = fig.add_axes(inset_rect, projection=ccrs.PlateCarree())
        ax.set_extent([124.0, 132.5, 33.0, 39.5], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eef4fb")
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e5ecdf")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
        ax.plot([lon], [lat], marker="o", markersize=5, markeredgecolor="black", markerfacecolor="#d7263d", transform=ccrs.PlateCarree(), zorder=5)
        ax.text(
            lon + 0.2,
            lat + 0.08,
            "Ulsan",
            transform=ccrs.PlateCarree(),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="black", linewidth=0.5),
        )
        ax.set_title("South Korea", fontsize=8.5, pad=2)
    except Exception:
        ax = fig.add_axes(inset_rect)
        ax.set_axis_off()
        ax.add_patch(
            FancyBboxPatch(
                (0, 0),
                1,
                1,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                transform=ax.transAxes,
                facecolor="#f7f9fc",
                edgecolor="#c8d2e1",
                linewidth=1.0,
            )
        )
        ax.text(0.5, 0.58, "Inset map unavailable\n(install cartopy)", ha="center", va="center", fontsize=8)
        ax.text(0.5, 0.23, f"Ulsan: {lat:.3f}°N, {lon:.3f}°E", ha="center", va="center", fontsize=7.5)


def build_figure(
    input_image: Path,
    output_base: Path,
    site_px: Tuple[int, int],
    site_label: str,
    dpi: int,
    enhance: bool,
) -> None:
    image = Image.open(input_image).convert("RGB")
    if enhance:
        image = ImageEnhance.Contrast(image).enhance(1.10)
        image = ImageEnhance.Sharpness(image).enhance(1.12)

    arr = np.asarray(image)
    h, w = arr.shape[:2]

    set_style()
    fig = plt.figure(figsize=(13.5, 8.0), constrained_layout=False)

    # Main image panel
    ax_img = fig.add_axes([0.04, 0.12, 0.58, 0.80])
    ax_img.imshow(arr)
    ax_img.set_axis_off()
    draw_marker(ax_img, site_px[0], site_px[1], site_label, w, h)
    draw_north_arrow(ax_img)

    ax_img.text(
        0.02,
        0.03,
        "Scale bar omitted (non-georeferenced preview)",
        transform=ax_img.transAxes,
        fontsize=8.4,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=0.6),
    )

    # Inset map
    add_inset_map(fig, lon=129.1269, lat=35.56233)

    # Title block
    fig.text(
        0.04,
        0.95,
        "Site Description and Dataset Overview (Yongyeon WWTP, Ulsan, South Korea)",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#10233f",
    )
    fig.text(
        0.04,
        0.915,
        "Satellite preview annotated for study context; operational metadata summarized for Section 2.1",
        ha="left",
        va="center",
        fontsize=10,
        color="#3e4a59",
    )

    # Right-column cards
    ax_card1 = fig.add_axes([0.66, 0.62, 0.31, 0.28])
    card1 = (
        "• Location: Seosaeng-myeon, Ulju-gun, Ulsan\n"
        "  (35.56233°N, 129.1269°E)\n"
        "• Large municipal WWTP supporting\n"
        "  urban + industrial wastewater treatment\n"
        "• Typical daily inflow capacity:\n"
        "  ~200,000–250,000 m³/day\n"
        "• Co-treatment includes municipal\n"
        "  wastewater and food waste"
    )
    draw_card(ax_card1, "Facility Context", card1)

    ax_card2 = fig.add_axes([0.66, 0.34, 0.31, 0.24])
    card2 = (
        "• Monitoring period: 2021-01-04 to 2023-10-31\n"
        "• Sampling frequency: Daily aggregates\n"
        "• Approximate sample size: n ≈ 1,031 days\n"
        "• Interpolated entries noted around\n"
        "  2023-07-25 to 2023-07-28"
    )
    draw_card(ax_card2, "Dataset Scope", card2)

    ax_card3 = fig.add_axes([0.66, 0.11, 0.31, 0.20])
    card3 = (
        "Influent/Effluent variables:\n"
        "Inflow/Outflow (m³/day), BOD, TOC, SS,\n"
        "TN, TP (mg/L), Coliform (MPN/100 mL).\n"
        "\n"
        "TMS compliance monitoring enables\n"
        "automated real-time reporting of key\n"
        "effluent indicators (TOC, TN, TP)\n"
        "to K-eco for regulatory quality control."
    )
    draw_card(ax_card3, "Measurements and Compliance (TMS)", card3)

    # Footer citation text
    fig.text(
        0.04,
        0.045,
        "Image source context: NASA Harmonized Landsat/Sentinel-2 (HLSL30 v2.0; DOI: 10.5067/HLS/HLSL30.002).",
        ha="left",
        va="center",
        fontsize=8.4,
        color="#4a5563",
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Section 2.1 site + dataset composite figure")
    parser.add_argument("--input", type=Path, required=True, help="Input satellite image (JPG/PNG)")
    parser.add_argument("--output", type=Path, required=True, help="Output basename (without extension)")
    parser.add_argument("--site-px", type=int, nargs=2, default=[722, 270], metavar=("X", "Y"), help="Site marker pixel coordinates")
    parser.add_argument("--site-label", type=str, default="Yongyeon WWTP", help="Site label text")
    parser.add_argument("--dpi", type=int, default=600, help="Output DPI")
    parser.add_argument("--enhance", action="store_true", help="Apply mild contrast/sharpness enhancement")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    img = Image.open(args.input)
    w, h = img.size
    x, y = int(args.site_px[0]), int(args.site_px[1])
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"site-px {(x, y)} outside bounds {(w, h)}")

    build_figure(
        input_image=args.input,
        output_base=args.output,
        site_px=(x, y),
        site_label=args.site_label,
        dpi=int(args.dpi),
        enhance=bool(args.enhance),
    )


if __name__ == "__main__":
    main()
