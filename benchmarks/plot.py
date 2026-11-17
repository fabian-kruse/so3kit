"""Plot benchmarks/results/*.csv into benchmarks/benchmarks.png (README figure).

Each panel shows one fixed workload and, per library, its strongest variant that was checked for
correctness (see the README "Benchmarks" section). so3kit is shown at its default precision, not the
faster float32 output path, so the ratios are not inflated.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).parent
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
# Colour follows the library across panels; each line also has its own marker and a direct label.
LIB = {
    "so3kit": "#2a78d6",
    "J-matrix": "#eb6834",
    "e3nn": "#1baf7a",
    "s2fft": "#e87ba4",
}

# (csv backend name, label, library, marker, linestyle)
PANELS = [
    dict(file="wigner.csv", x="ell_max", y="rotations_per_s", where={"n": "10000"},
         title="Wigner-D matrices (10,000 rotations)", xlabel="bandlimit ℓ_max",
         ylabel="rotations / s (higher is better)",
         series=[("so3kit", "so3kit", "so3kit", "o", "-"),
                 ("J-matrix", "J-matrix", "J-matrix", "s", "-"),
                 ("e3nn", "e3nn", "e3nn", "^", "-")]),
    dict(file="sofft.csv", x="L", y="seconds_per_function", invert=True, where={"batch": "1"},
         title="Inverse SO(3) FFT (1 function)", xlabel="bandlimit L",
         ylabel="functions / s (higher is better)",
         series=[("so3kit", "so3kit", "so3kit", "o", "-"),
                 ("s2fft precomputed kernel [real]", "s2fft precomputed", "s2fft", "D", "-"),
                 ("s2fft recursive [real]", "s2fft recursive", "s2fft", "P", "--")]),
    dict(file="derivatives.csv", x="ell_max", y="points_per_s", where={"n": "10000"},
         title="Gradient + Hessian (10,000 points)", xlabel="bandlimit ℓ_max",
         ylabel="points / s (higher is better)",
         series=[("so3kit", "so3kit", "so3kit", "o", "-"),
                 ("autodiff J-matrix (batched reverse)", "autodiff J-matrix", "J-matrix", "s", "-"),
                 ("autodiff e3nn (batched reverse)", "autodiff e3nn", "e3nn", "^", "-")]),
]


def load(name):
    path = HERE / "results" / name
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def key(r):
    """Backend name; s2fft rows carry their signal type, e.g. 's2fft precomputed kernel [real]'."""
    if r.get("signal") and r["backend"] != "so3kit":
        return f'{r["backend"]} [{r["signal"]}]'
    return r["backend"]


def panel(ax, spec):
    rows = [r for r in load(spec["file"]) if all(r.get(k) == v for k, v in spec["where"].items())]
    ax.set_title(spec["title"], loc="left", fontsize=10, color=INK)
    xticks = sorted({float(r[spec["x"]]) for r in rows})
    for backend, label, lib, marker, ls in spec["series"]:
        f = (lambda v: 1.0 / v) if spec.get("invert") else (lambda v: v)
        pts = sorted((float(r[spec["x"]]), f(float(r[spec["y"]]))) for r in rows if key(r) == backend)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=LIB[lib], marker=marker, ls=ls, lw=2, ms=7,
                markeredgecolor="white", markeredgewidth=1)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yscale("log")
    ax.set_xlabel(spec["xlabel"], fontsize=8, color=MUTED)
    ax.set_ylabel(spec["ylabel"], fontsize=8, color=MUTED)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.margins(x=0.08)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 2.2)  # room for the direct labels
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main(gpu="NVIDIA GeForce RTX 4090"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, spec in zip(axes, PANELS):
        panel(ax, spec)
    handles = [Line2D([], [], color=c, lw=2, label=lib) for lib, c in LIB.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, f"GPU: {gpu} · single precision baselines · setup excluded, reported in the CSVs",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(HERE / "benchmarks.png", dpi=200)
    print(f"wrote {HERE / 'benchmarks.png'}")


if __name__ == "__main__":
    main()
