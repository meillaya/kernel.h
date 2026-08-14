#!/usr/bin/env python3
"""Performance charts + measured roofline (plan todo 38).

Reads a benchmark results CSV (G32 schema, plan todo 35):
    kernel,shape,backend,ms_median,gflops_or_gbps,timestamp,device
and writes, into an output directory:

  gemm_gflops_vs_size.png   GEMM GFLOPS vs N (log x) per backend.
  elementwise_bw.png        effective GB/s grouped bars for
                            softmax / layernorm / scan / fused_gelu.
  attention_gflops.png      attention GFLOPS per sequence length.
  roofline.png              Williams et al. roofline with MEASURED ceilings.
  charts_stats.csv          machine-readable summary (asserted by tests).

Arithmetic intensity follows scripts/README.md (todo 35 owns the formulas):

  GEMM       AI = 2mnk / ((mk + kn + mn) * 4)
  attention  AI = s / 4   (4bh s^2 d flops / (4 bh s d * 4) bytes)
  softmax / layernorm / scan / fused_gelu: AI ~ 0 (no flops counted; they are
             bandwidth-bound rows -> plotted as effective GB/s in
             elementwise_bw.png, NOT as roofline AI points; their roofline
             ceiling is roofline_triad).

Roofline ceilings are MEASURED rows from the CSV (never assumed):
  roofline_triad  -> effective bandwidth ceiling (GB/s)
  roofline_fma    -> compute ceiling (GFLOPS)
Ridge point at AI = compute_ceiling / bandwidth_ceiling
(Williams et al., "Roofline: an insightful visual performance model for
multicore architectures", doi:10.1145/1498765.1498785).

Degraded mode: charts are produced from whatever rows exist. If the roofline
ceiling rows are missing the other three charts are still produced with a
title note; if no local-execution backend rows exist the titles carry
"local execution unavailable".

Usage:
    python3 scripts/charts.py [--csv results.csv] [--out out-dir]
    python3 scripts/charts.py results.csv out-dir          # positional form
Defaults: --csv build/bench/results.csv --out build/bench/charts.
Exit code is 0 on success; non-zero with a message when the CSV has no data
rows (e.g. header-only input) - no empty charts are silently produced.
"""

import argparse
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Backend labels in the order the GEMM chart draws them (taxonomy from the
# plan / todo 35; variants that have no rows for a kernel are skipped).
LOCAL_BACKENDS = ("kernelh", "rocblas", "hipref-naive", "hipref-vec")
GEMM_BACKEND_ORDER = ("kernelh", "rocblas", "hipref-naive", "hipref-vec",
                      "cublas", "cutlass")
BANDWIDTH_KERNELS = ("softmax", "layernorm", "scan", "fused_gelu")

# PNG size / dpi tuned so a 4-line chart clears the >10KB assertion.
FIGSIZE = (9.0, 5.5)
DPI = 150


def parse_shape(shape_text):
    """'k=v,k=v,...' -> {'k': 'v', ...}. Handles the quoted CSV cells."""
    result = {}
    for part in shape_text.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2:
            result[kv[0].strip()] = kv[1].strip()
    return result


def shape_int(shape, key, default=0):
    try:
        return int(shape[key])
    except (KeyError, ValueError):
        return default


def compute_ai(kernel, shape):
    """Arithmetic intensity (flops/byte) per scripts/README.md.

    Returns None for kernels with no documented FLOPs count (bandwidth-bound
    rows) or when the shape params needed by the formula are missing.
    """
    if kernel == "gemm":
        m = shape_int(shape, "m")
        n = shape_int(shape, "n")
        k = shape_int(shape, "k")
        if m <= 0 or n <= 0 or k <= 0:
            return None
        # AI = 2mnk / ((mk + kn + mn) * 4)   [fp32: 4 bytes/element]
        return (2.0 * m * n * k) / ((m * k + k * n + m * n) * 4.0)
    if kernel == "attention":
        s = shape_int(shape, "seq")
        if s <= 0:
            return None
        # AI = 4bh s^2 d / (4 bh s d * 4) = s / 4
        return s / 4.0
    # softmax / layernorm / scan / fused_gelu: no flops counted (README) ->
    # AI ~ 0; they are plotted as GB/s, not as roofline AI points.
    if kernel in BANDWIDTH_KERNELS:
        return 0.0
    return None


def load_rows(csv_path):
    """Returns the parsed CSV rows as a list of dicts (G32 schema)."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"kernel", "shape", "backend", "ms_median", "gflops_or_gbps"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "results.csv is missing required G32 columns; got header: "
                f"{reader.fieldnames}")
        return [row for row in reader]


def row_float(row, column):
    try:
        value = float(row[column])
    except (KeyError, ValueError, TypeError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def ceiling_from_rows(rows):
    """Measured ceilings from the roofline_triad / roofline_fma rows."""
    bw_gbps = None
    peak_gflops = None
    for row in rows:
        rate = row_float(row, "gflops_or_gbps")
        if rate is None:
            continue
        if row["kernel"] == "roofline_triad":
            bw_gbps = rate
        elif row["kernel"] == "roofline_fma":
            peak_gflops = rate
    return bw_gbps, peak_gflops


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def add_degraded_note(title, local_available, note_ceilings=False):
    """Appends the degraded-mode notes the plan pins to chart titles."""
    notes = []
    if not local_available:
        notes.append("local execution unavailable")
    if note_ceilings:
        notes.append("measured roofline ceilings unavailable")
    if notes:
        return title + "  [" + "; ".join(notes) + "]"
    return title


def chart_gemm(rows, out_dir, local_available):
    """gemm_gflops_vs_size.png: GFLOPS vs N (log x) per backend."""
    data = {backend: [] for backend in GEMM_BACKEND_ORDER}
    for row in rows:
        if row["kernel"] != "gemm":
            continue
        rate = row_float(row, "gflops_or_gbps")
        if rate is None:
            continue
        n = shape_int(parse_shape(row["shape"]), "n")
        if n <= 0:
            continue
        if row["backend"] in data:
            data[row["backend"]].append((n, rate))
    for backend in data:
        data[backend].sort()

    if not any(data.values()):
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for backend, points in data.items():
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.loglog(xs, ys, marker="o", label=backend)
    ax.set_xlabel("N (square GEMM, M=K=N)")
    ax.set_ylabel("GFLOPS (2N^3 / t)")
    ax.set_title(add_degraded_note(
        "GEMM performance vs problem size (log-log)", local_available))
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save(fig, out_dir, "gemm_gflops_vs_size.png")
    return True


def chart_elementwise(rows, out_dir, local_available):
    """elementwise_bw.png: effective GB/s grouped bars per kernel@shape."""
    # One bar group per (kernel, shape); one bar per backend with data.
    groups = []  # list of (label, [(backend, gbps), ...])
    grouped = {}
    for row in rows:
        if row["kernel"] not in BANDWIDTH_KERNELS:
            continue
        rate = row_float(row, "gflops_or_gbps")
        if rate is None:
            continue
        key = (row["kernel"], row["shape"])
        grouped.setdefault(key, []).append((row["backend"], rate))
    for (kernel, shape) in sorted(grouped):
        bars = grouped[(kernel, shape)]
        label = f"{kernel}\n({shape})"
        groups.append((label, sorted(bars)))

    if not groups:
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    n_groups = len(groups)
    width = 0.8 / max(len(bars) for _, bars in groups)
    for gi, (label, bars) in enumerate(groups):
        n_bars = len(bars)
        offset = -((n_bars - 1) / 2.0) * width
        for bi, (backend, gbps) in enumerate(bars):
            x = gi + offset + bi * width
            ax.bar(x, gbps, width=width, label=backend)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # dedupe backend legend entries
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7)
    ax.set_ylabel("effective GB/s")
    ax.set_title(add_degraded_note(
        "Elementwise / reduction bandwidth (grouped by kernel@shape)",
        local_available))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, out_dir, "elementwise_bw.png")
    return True


def chart_attention(rows, out_dir, local_available):
    """attention_gflops.png: attention GFLOPS per sequence length."""
    points = []
    for row in rows:
        if row["kernel"] != "attention":
            continue
        rate = row_float(row, "gflops_or_gbps")
        if rate is None:
            continue
        shape = parse_shape(row["shape"])
        s = shape_int(shape, "seq")
        if s <= 0:
            continue
        points.append((s, row["backend"], rate))
    if not points:
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # Group bars by seq length, one bar per backend.
    seqs = sorted({p[0] for p in points})
    by_backend = {}
    for s, backend, rate in points:
        by_backend.setdefault(backend, []).append((s, rate))
    width = 0.8 / len(by_backend)
    for bi, (backend, pts) in enumerate(sorted(by_backend.items())):
        pts.sort()
        xs = [seqs.index(s) + (bi - (len(by_backend) - 1) / 2.0) * width
              for s, _ in pts]
        ys = [rate for _, rate in pts]
        ax.bar(xs, ys, width=width, label=backend)
    ax.legend()
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels([f"seq={s}" for s in seqs])
    ax.set_ylabel("GFLOPS")
    ax.set_title(add_degraded_note(
        "Attention performance (b=2 h=8 head_dim=64)", local_available))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, out_dir, "attention_gflops.png")
    return True


def chart_roofline(rows, out_dir, local_available, bw_gbps, peak_gflops):
    """roofline.png: Williams et al. roofline with measured ceilings."""
    # Kernel AI points: GEMM (AI = 2mnk/((mk+kn+mn)*4)) and attention
    # (AI = s/4). Bandwidth-bound kernels are NOT AI points (README).
    points = []
    for row in rows:
        if row["kernel"] not in ("gemm", "attention"):
            continue
        rate = row_float(row, "gflops_or_gbps")
        if rate is None:
            continue
        ai = compute_ai(row["kernel"], parse_shape(row["shape"]))
        if ai is None or ai <= 0.0:
            continue
        points.append((ai, rate, row["backend"], row["kernel"]))

    if bw_gbps is None or peak_gflops is None:
        # Degraded: no measured ceilings. Still draw the kernel points if any,
        # with the title note (plan: "if roofline rows missing, still produce
        # the other 3 charts and note in the title").
        if not points:
            return False
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for ai, rate, backend, kernel in points:
            ax.scatter([ai], [rate], s=40)
            ax.annotate(f"{kernel}\n{backend}", (ai, rate), fontsize=6,
                        textcoords="offset points", xytext=(4, 4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic intensity AI (flops/byte)")
        ax.set_ylabel("GFLOPS")
        ax.set_title(add_degraded_note(
            "Roofline (measured ceilings unavailable)", local_available,
            note_ceilings=True))
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        save(fig, out_dir, "roofline.png")
        return True

    ridge = peak_gflops / bw_gbps  # flops/byte at the ridge point
    min_x = min([p[0] for p in points] + [ridge])
    max_x = max([p[0] for p in points] + [ridge])
    x_lo = max(min_x / 20.0, ridge / 100.0)
    x_hi = max(max_x * 10.0, ridge * 10.0)

    # Sample the memory-bound line y = bw * x (slope-1 in log-log) and the
    # compute-bound ceiling y = peak above the ridge.
    xs = []
    ys = []
    steps = 400
    for i in range(steps + 1):
        x = x_lo * (x_hi / x_lo) ** (i / steps)
        xs.append(x)
        if x < ridge:
            ys.append(bw_gbps * x)  # memory-bound region
        else:
            ys.append(peak_gflops)  # compute-bound region

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(xs, ys, color="black", lw=2,
              label=f"roofline (ridge AI={ridge:.1f})")
    ax.scatter([ridge], [peak_gflops], color="red", marker="x", s=80,
               label=f"ridge ({ridge:.1f} flops/byte, {peak_gflops:.0f} GFLOPS)")
    for ai, rate, backend, kernel in points:
        ax.scatter([ai], [rate], s=45)
        ax.annotate(f"{kernel}\n{backend}", (ai, rate), fontsize=6,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity AI (flops/byte)")
    ax.set_ylabel("GFLOPS")
    ax.set_title(add_degraded_note(
        "Roofline with measured ceilings "
        f"(BW={bw_gbps:.1f} GB/s, peak={peak_gflops:.0f} GFLOPS)",
        local_available))
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save(fig, out_dir, "roofline.png")
    return True


def write_charts_stats(rows, out_path, bw_gbps, peak_gflops):
    """charts_stats.csv: machine-readable per-row summary (tested)."""
    ridge = peak_gflops / bw_gbps if (bw_gbps and peak_gflops) else None
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "backend", "shape", "ms_median",
                         "gflops_or_gbps", "ai", "ceiling_ai",
                         "ceiling_gflops"])
        written = 0
        for row in rows:
            if row["kernel"] in ("roofline_triad", "roofline_fma"):
                # The ceilings themselves are not kernel measurements.
                continue
            ms = row_float(row, "ms_median")
            rate = row_float(row, "gflops_or_gbps")
            ai = compute_ai(row["kernel"], parse_shape(row["shape"]))
            writer.writerow([
                row["kernel"], row["backend"], row["shape"],
                "" if ms is None else f"{ms:.6f}",
                "" if rate is None else f"{rate:.6f}",
                "" if ai is None else f"{ai:.6f}",
                "" if ridge is None else f"{ridge:.6f}",
                "" if peak_gflops is None else f"{peak_gflops:.6f}",
            ])
            written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="kernel.h performance charts + measured roofline (todo 38)")
    parser.add_argument("--csv", default="build/bench/results.csv",
                        help="results CSV (G32 schema); default "
                             "build/bench/results.csv")
    parser.add_argument("--out", default="build/bench/charts",
                        help="output directory for the PNGs + charts_stats.csv; "
                             "default build/bench/charts")
    parser.add_argument("positional", nargs="*",
                        help="[results.csv [out-dir]] positional form")
    args = parser.parse_args(argv)

    csv_path = args.csv
    out_dir = args.out
    if args.positional:
        csv_path = args.positional[0]
        if len(args.positional) > 1:
            out_dir = args.positional[1]

    if not os.path.isfile(csv_path):
        print(f"charts.py: results CSV not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        rows = load_rows(csv_path)
    except (OSError, ValueError) as exc:
        print(f"charts.py: cannot read results CSV: {exc}", file=sys.stderr)
        return 1

    # QA scenario (plan): header-only / empty CSV -> non-zero + message, never
    # empty charts.
    if not rows:
        print("charts.py: no data rows in "
              f"{csv_path}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)

    bw_gbps, peak_gflops = ceiling_from_rows(rows)
    local_available = any(r["backend"] in LOCAL_BACKENDS for r in rows)

    produced = [
        chart_gemm(rows, out_dir, local_available),
        chart_elementwise(rows, out_dir, local_available),
        chart_attention(rows, out_dir, local_available),
        chart_roofline(rows, out_dir, local_available, bw_gbps, peak_gflops),
    ]

    stats_path = os.path.join(out_dir, "charts_stats.csv")
    n_stats = write_charts_stats(rows, stats_path, bw_gbps, peak_gflops)

    print(f"charts.py: wrote {sum(produced)} charts to {out_dir} "
          f"(charts_stats.csv: {n_stats} rows)")
    if not all(produced):
        missing = [name for name, ok in
                   zip(["gemm_gflops_vs_size", "elementwise_bw",
                        "attention_gflops", "roofline"], produced) if not ok]
        print(f"charts.py: warning: no data for chart(s): {missing}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
