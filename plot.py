#!/usr/bin/env python
"""Plot the experiment runs of train.sh: time, memory and error of every run, the plans
the optimizer chose, and how well its models predicted them.

    python plot.py                                    # stats/experiments -> stats/experiments/plots/*.png
    python plot.py stats/experiments --reference base_fp32 --out plots --format pdf
    python plot.py stats/experiments --only tradeoff sweeps --theme dark

The inputs are the JSON logs of apps/train_policies.py, read the same way as
apps/summarize_runs.py (saved profiles and .partial logs are skipped), and the numbers
are the same: ms/step is the median over the epochs, peak memory the largest epoch.
Runs are grouped by name: a fixed policy (--static, e.g. base_*), an ablation (abl_*)
or an LP optimizer run (the rest). The figures:

    tradeoff    step time and test perplexity against peak memory, one point per run
    overview    every run: ms/step, peak memory, test perplexity vs the reference,
                planning time
    sweeps      the LP runs along the memory budget (at the most common eps) and along
                the error limit (at the most common budget), with the fixed policies as
                reference lines
    loss        training loss relative to the reference per epoch, one panel per run
    plans       the policy of every layer in every epoch, one panel per optimizer run
    replanning  what the re-plans measured in the run with the most of them: the batch
                noise, the fp16 error of every layer, the predicted error of the plan
    model       measured vs predicted peak memory and step time of every plan

Needs matplotlib (pip install matplotlib).
"""
import argparse
import os
import statistics
import sys
import textwrap
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apps.summarize_runs import MIB, compare, load, mode, stats  # noqa: E402

GIB = 2 ** 30
FIGURES = ["tradeoff", "overview", "sweeps", "loss", "plans", "replanning", "model"]

# The reference palette of the dataviz method: chart chrome, ink and the categorical
# slots (fixed order, light and dark steps of the same hues).
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9",
                  baseline="#c3c2b7", context="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", secondary="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 baseline="#383835", context="#52514e",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}
# run groups -> categorical slots 1-3 (the most a scatter may use)
GROUPS = [("fixed", "fixed policy (no LP)"), ("lp", "LP optimizer"), ("ablation", "LP ablation")]
# a policy is two choices: hue = precision (blue fp32, orange fp16), a light tint = the
# activations are recomputed in backward instead of stored (faded toward the surface)
POLICIES = [("fp32", "fp32: store", 0, False), ("none", "none: recompute", 0, True),
            ("fp16", "fp16: store in fp16", 1, False), ("fp16_none", "fp16_none: fp16 + recompute", 1, True)]


def tint(color, surface, amount=0.38):
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(surface[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(amount * x + (1 - amount) * y):02x}" for x, y in zip(a, b))


class Context:
    def __init__(self, runs, ref, theme):
        self.runs, self.ref, self.t = runs, ref, THEMES[theme]
        self.color = {g: self.t["series"][i] for i, (g, _) in enumerate(GROUPS)}

    def by_group(self, *groups):
        return [r for r in self.runs if r["group"] in groups]


# ---------------------------------------------------------------------------- data

def group_of(name, log):
    if log["config"].get("static_policies"):
        return "fixed"
    return "ablation" if name.startswith("abl") else "lp"


def collect(logs, reference):
    rows = {name: stats(log) for name, log in logs.items()}
    ref = rows.get(reference)
    for name, row in rows.items():
        compare(row, ref)
        log = logs[name]
        row.update(name=name, log=log, group=group_of(name, log),
                   infeasible=any(not h["feasible"] for h in log.get("replans", [])))
    order = [g for g, _ in GROUPS]
    runs = sorted(rows.values(), key=lambda r: (r["name"] != reference, order.index(r["group"]), r["name"]))
    return runs, ref


def gib(row):
    return None if row["peak_mib"] is None else row["peak_mib"] * MIB / GIB


def pct(v):
    return None if v is None else 100 * v


def run_mode(row):
    """summarize_runs.mode without the "LP " prefix; a re-plan period past the last epoch is "planned once"."""
    text = mode(row["log"])
    k = row["log"]["config"].get("replan_every")
    if k is not None and k >= len(row["log"]["epochs"]) > 1:
        text = text.replace(f"k={k}", "planned once")
    return text[3:] if text.startswith("LP ") else text


def label(row):
    return row["name"] + (" (infeasible)" if row["infeasible"] else "")


# --------------------------------------------------------------------------- style

def setup(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["baseline"], "axes.linewidth": 0.8, "axes.labelcolor": t["secondary"],
        "axes.labelsize": 8.5, "axes.titlesize": 9.5, "axes.titlecolor": t["ink"], "axes.titlelocation": "left",
        "axes.titlepad": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": False, "grid.color": t["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": t["baseline"], "ytick.color": t["baseline"], "xtick.labelcolor": t["muted"],
        "ytick.labelcolor": t["muted"], "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 3, "ytick.major.size": 3, "font.size": 8.5, "text.color": t["ink"],
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": t["secondary"],
    })
    return plt


def header(fig, t, title, subtitle=None, handles=None):
    """Title, subtitle and the legend in a band above the panels (left-aligned)."""
    w, h = fig.get_size_inches()
    lines = textwrap.wrap(subtitle, int(w * 15)) if subtitle else []
    band = 0.42 + 0.19 * len(lines) + (0.1 if lines else 0) + (0.34 if handles else 0)
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - band / h))
    x = 0.14 / w
    fig.text(x, 1 - 0.14 / h, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=t["ink"])
    if lines:
        fig.text(x, 1 - 0.44 / h, "\n".join(lines), ha="left", va="top", fontsize=9, color=t["secondary"],
                 linespacing=1.35)
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x - 0.06 / w, 1 - (band - 0.36) / h),
                   ncol=len(handles), handlelength=1.4, columnspacing=1.8, borderaxespad=0, borderpad=0)


def group_handles(ctx, groups, infeasible=False, marker="o"):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=marker, ls="none", ms=7, color=ctx.color[g], mec=ctx.t["surface"], label=lbl)
          for g, lbl in GROUPS if g in groups]
    if infeasible:
        hs.append(Line2D([], [], marker=marker, ls="none", ms=7, mfc=ctx.t["surface"], mec=ctx.t["muted"], mew=1.6,
                         label="hollow: no plan met every limit"))
    return hs


def grid(ax, axis="y"):
    ax.grid(True, axis=axis)


def finalize(fig, placers):
    """Labels that depend on the final layout: place them, let the layout make room,
    place them again on the settled axes, then freeze the layout."""
    fig.canvas.draw()
    artists = [a for p in placers for a in p()]
    fig.canvas.draw()
    for a in artists:
        a.remove()
    for p in placers:
        p()
    fig.set_layout_engine("none")


def point_labels(fig, ax, items, t, size=7.5):
    """items [(x, y, text)]: a label beside every point, at the first spot that is inside
    the axes and clear of the other labels and points; farther spots get a leader line."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    def placer():
        r = fig.canvas.get_renderer()
        box = ax.get_window_extent(r)
        px = fig.dpi / 72
        taken = []
        for x, y, _ in items:
            X, Y = ax.transData.transform((x, y))
            taken.append(Bbox.from_extents(X - 4 * px, Y - 4 * px, X + 4 * px, Y + 4 * px))  # the dot
        made = []
        for x, y, text in items:
            spots = [(d, dx, dy) for d in (7, 14, 22, 32) for dx, dy in
                     ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))]
            for d, dx, dy in spots:
                a = ax.annotate(text, (x, y), xytext=(dx * d, dy * d), textcoords="offset points", fontsize=size,
                                color=t["secondary"], ha="left" if dx > 0 else "right" if dx < 0 else "center",
                                va="bottom" if dy > 0 else "top" if dy < 0 else "center", linespacing=1.1,
                                arrowprops=None if d == 7 else dict(arrowstyle="-", color=t["muted"], lw=0.6,
                                                                    shrinkA=1, shrinkB=4))
                a.update_positions(r)  # the text is placed at draw time
                bb = Text.get_window_extent(a, r).padded(1 * px)
                inside = box.x0 <= bb.x0 and bb.x1 <= box.x1 and box.y0 <= bb.y0 and bb.y1 <= box.y1
                if inside and not any(bb.overlaps(o) for o in taken):
                    taken.append(bb)
                    made.append(a)
                    break
                a.remove()
            else:
                made.append(ax.annotate(text, (x, y), xytext=(7, 0), textcoords="offset points", fontsize=size,
                                        color=t["secondary"], va="center"))
        return made
    return placer


def end_labels(fig, ax, items, t, size=7.5):
    """items [(y, text)]: labels right of the axes at data y (reference lines), pushed
    apart where they would collide, each tied to its line by a leader."""
    def placer():
        if not items:
            return []
        gap = size * 1.3 * fig.dpi / 72
        ys = sorted((ax.transData.transform((0, y))[1], y, text) for y, text in items)
        pos = [ys[0][0]]
        for Y, _, _ in ys[1:]:
            pos.append(max(Y, pos[-1] + gap))
        shift = sum(p - Y for p, (Y, _, _) in zip(pos, ys)) / len(ys)
        made = []
        for p, (Y, y, text) in zip(pos, ys):
            made.append(ax.annotate(text, (1, y), xycoords=ax.get_yaxis_transform(),
                                    xytext=(12, (p - shift - Y) * 72 / fig.dpi), textcoords="offset points",
                                    ha="left", va="center", fontsize=size, color=t["secondary"],
                                    annotation_clip=False,
                                    arrowprops=dict(arrowstyle="-", color=t["context"], lw=0.8, shrinkA=0, shrinkB=1)))
        return made
    return placer


def merge_points(points, tol):
    """[(x, y, name)] -> [(x, y, "name\\nname")]: points closer than tol (relative) share a label."""
    merged = []
    for x, y, name in points:
        for m in merged:
            if abs(m[0] - x) <= tol[0] and abs(m[1] - y) <= tol[1]:
                m[2].append(name)
                break
        else:
            merged.append([x, y, [name]])
    return [(x, y, "\n".join(names)) for x, y, names in merged]


def scatter_runs(ax, ctx, pts):
    """pts [(x, y, row)]: one dot per run, colored by group, hollow when a plan was infeasible."""
    t = ctx.t
    for g, _ in GROUPS:
        for hollow in (False, True):
            sel = [(x, y) for x, y, r in pts if r["group"] == g and r["infeasible"] == hollow]
            if not sel:
                continue
            xs, ys = zip(*sel)
            if hollow:  # a ring, larger than a dot and below the dots, so a run at the same spot shows inside
                ax.scatter(xs, ys, s=100, facecolors=t["surface"], edgecolors=ctx.color[g], linewidths=1.8, zorder=2.5)
            else:
                ax.scatter(xs, ys, s=64, color=ctx.color[g], edgecolors=t["surface"], linewidths=1.6, zorder=3)


# ------------------------------------------------------------------------- figures

def fig_tradeoff(plt, ctx):
    runs = [r for r in ctx.runs if r["peak_mib"] is not None and r["ms_step"]]
    if len(runs) < 2:
        return None
    t, ref = ctx.t, ctx.ref
    rel = ref is not None and ref.get("test_ppl") is not None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
    second = (lambda r: pct(r["d_ppl"])) if rel else (lambda r: r["test_ppl"])
    specs = [(lambda r: r["ms_step"], "ms per training step (median over the epochs)",
              "Step time against peak memory", ref and ref["ms_step"]),
             (second, "test perplexity vs " + ref["name"] + " (%)" if rel else "test perplexity",
              "Test perplexity against peak memory", 0 if rel else None)]
    placers = []
    for ax, (get, ylabel, title, ref_y) in zip(axes, specs):
        pts = [(gib(r), get(r), r) for r in runs if get(r) is not None]
        if ref is not None:
            ax.axvline(gib(ref), color=t["baseline"], lw=0.8, zorder=1)
        if ref_y is not None:
            ax.axhline(ref_y, color=t["baseline"], lw=0.8, zorder=1)
        scatter_runs(ax, ctx, pts)
        grid(ax, "both")
        ax.set_xlabel("peak memory (GiB, needle's counter)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.margins(x=0.12, y=0.14)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        tol = (0.012 * (max(xs) - min(xs) or 1), 0.012 * (max(ys) - min(ys) or 1))
        placers.append(point_labels(fig, ax, merge_points([(x, y, r["name"]) for x, y, r in pts], tol), t))
    sub = "One dot per run. Lower is better on both panels"
    if ref is not None:
        sub += f"; the hairlines cross at {ref['name']}, the reference ({gib(ref):.1f} GiB, {ref['ms_step']:.0f} ms/step"
        sub += f", test perplexity {ref['test_ppl']:.1f})." if rel else ")."
    header(fig, t, "Time, memory and error of every run", sub,
           group_handles(ctx, {r["group"] for r in runs}, any(r["infeasible"] for r in runs)))
    finalize(fig, placers)
    return fig


def bars(ax, ctx, rows, values, fmt, zero_centered=False):
    t = ctx.t
    ys = list(range(len(rows)))
    vals = [v if v is not None else 0 for v in values]
    ax.barh(ys, vals, height=0.56, color=[ctx.color[r["group"]] for r in rows], zorder=2)
    span = max((abs(v) for v in vals), default=1) or 1
    for y, v, raw in zip(ys, vals, values):
        if raw is None:
            ax.text(0, y, " –", va="center", ha="left", fontsize=7.5, color=t["muted"])
            continue
        off = 0.02 * span * (1 if v >= 0 else -1)
        ax.text(v + off, y, fmt(raw), va="center", ha="left" if v >= 0 else "right", fontsize=7.5, zorder=4,
                color=t["secondary"], bbox=dict(facecolor=t["surface"], edgecolor="none", pad=0.8))
    if zero_centered:
        ax.set_xlim(-1.45 * span, 1.45 * span)
    else:
        ax.set_xlim(0, 1.2 * span)
    ax.axvline(0, color=t["baseline"], lw=0.8, zorder=3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    grid(ax, "x")


def fig_overview(plt, ctx):
    rows = ctx.runs
    if not rows:
        return None
    t, ref = ctx.t, ctx.ref
    rel = ref is not None and ref.get("test_ppl") is not None
    fig, axes = plt.subplots(1, 4, figsize=(14, max(3.4, 1.5 + 0.34 * len(rows))), sharey=True, layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.8]))
    panels = [("ms per step", [r["ms_step"] for r in rows], lambda v: f"{v:.0f}", False,
               ref and ref["ms_step"]),
              ("peak memory (GiB)", [gib(r) for r in rows], lambda v: f"{v:.1f}", False, ref and gib(ref)),
              ("test perplexity vs " + ref["name"] + " (%)" if rel else "test perplexity",
               [pct(r["d_ppl"]) for r in rows] if rel else [r["test_ppl"] for r in rows],
               (lambda v: f"{v:+.1f}%") if rel else (lambda v: f"{v:.1f}"), rel, None),
              ("planning time (s)",
               [r["planning_s"] if r["replans"] else None for r in rows], lambda v: f"{v:.0f}", False, None)]
    for ax, (title, values, fmt, centered, ref_x) in zip(axes, panels):
        bars(ax, ctx, rows, values, fmt, centered)
        if ref_x:
            ax.axvline(ref_x, color=t["muted"], lw=0.8, zorder=3)
        ax.set_title(title)
    axes[0].set_yticks(range(len(rows)), [label(r) for r in rows])
    axes[0].tick_params(axis="y", labelcolor=t["secondary"])
    axes[0].invert_yaxis()
    sub = ("ms/step: median over the epochs; memory: the largest epoch; perplexity: the test set after training; "
           "planning: profiling + error estimates + solving, all re-plans.")
    if ref is not None:
        sub += f" Vertical rules: {ref['name']}."
    header(fig, t, "Every run at a glance", sub, group_handles(ctx, {r["group"] for r in rows}, marker="s"))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig


def fig_sweeps(plt, ctx):
    lp = ctx.by_group("lp")
    cfg = lambda r: r["log"]["config"]  # noqa: E731
    eps_ref = Counter(cfg(r).get("eps") for r in lp if cfg(r).get("eps") is not None).most_common(1)
    b_ref = Counter(cfg(r).get("budget_fraction") for r in lp
                    if cfg(r).get("budget_fraction") is not None).most_common(1)
    last = float("inf")
    sweeps = []
    if eps_ref:
        e = eps_ref[0][0]
        sel = sorted([r for r in lp if cfg(r).get("eps") == e and cfg(r).get("budget") is None
                      or cfg(r).get("eps") == e and cfg(r).get("budget_fraction") is not None],
                     key=lambda r: cfg(r).get("budget_fraction") or last)
        sweeps.append(("budget", f"Memory budget sweep at ε = {e:g}", sel,
                       [("no budget" if cfg(r).get("budget_fraction") is None else f"{cfg(r)['budget_fraction']:.0%}")
                        for r in sel], "memory budget (fraction of the fp32 peak)", e))
    if b_ref:
        b = b_ref[0][0]
        sel = sorted([r for r in lp if cfg(r).get("budget_fraction") == b],
                     key=lambda r: cfg(r).get("eps") if cfg(r).get("eps") is not None else last)
        sweeps.append(("eps", f"Error limit sweep at a budget of {b:.0%}", sel,
                       ["no limit" if cfg(r).get("eps") is None else f"{cfg(r)['eps']:g}" for r in sel],
                       "error limit ε (RMS gradient error / batch noise)", None))
    sweeps = [s for s in sweeps if len(s[2]) >= 2]
    if not sweeps:
        return None
    t, ref = ctx.t, ctx.ref
    rel = ref is not None and ref.get("test_ppl") is not None
    fixed = ctx.by_group("fixed")
    color = ctx.color["lp"]
    cols = [("ms per step", lambda r: r["ms_step"], "{:.0f}"),
            ("peak memory (GiB)", gib, "{:.1f}"),
            ("predicted error of the last plan", lambda r: r["pred_error"], None),
            ("test perplexity vs " + ref["name"] + " (%)" if rel else "test perplexity",
             (lambda r: pct(r["d_ppl"])) if rel else (lambda r: r["test_ppl"]), None)]
    fig, axes = plt.subplots(len(sweeps), len(cols), figsize=(15, 0.9 + 3.3 * len(sweeps)), layout="constrained",
                             squeeze=False)
    placers = []
    from matplotlib.lines import Line2D
    for row_axes, (kind, title, sel, ticks, xlabel, eps) in zip(axes, sweeps):
        xs = list(range(len(sel)))
        for j, (ax, (ylabel, get, _)) in enumerate(zip(row_axes, cols)):
            refs = []
            if j != 2:  # the fixed policies have no predicted error
                for r in fixed:
                    v = get(r)
                    if v is not None:
                        ax.axhline(v, color=t["context"], lw=1, zorder=1)
                        refs.append((v, r["name"].replace("base_", "")))
            if kind == "budget" and j == 1 and ref is not None:
                bud = [(x, cfg(r)["budget_fraction"] * gib(ref)) for x, r in zip(xs, sel)
                       if cfg(r).get("budget_fraction") is not None]
                if bud:
                    ax.plot(*zip(*bud), ls="none", marker="_", ms=22, mew=1.6, color=t["ink"], zorder=4)
            if j == 2:
                limits = [(x, cfg(r).get("eps")) for x, r in zip(xs, sel) if cfg(r).get("eps") is not None]
                if limits:
                    ax.plot(*zip(*limits), ls="none", marker="_", ms=22, mew=1.6, color=t["ink"], zorder=4)
            pts = [(x, get(r), r) for x, r in zip(xs, sel) if get(r) is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, zorder=2)
                for x, y, r in pts:
                    if r["infeasible"]:
                        ax.plot(x, y, marker="o", ms=8, mfc=t["surface"], mec=color, mew=1.8, zorder=3, clip_on=False)
                    else:
                        ax.plot(x, y, marker="o", ms=8.5, color=color, mec=t["surface"], mew=1.6, zorder=3,
                                clip_on=False)
            ax.set_xticks(xs, ticks)
            ax.set_xlim(-0.4, len(sel) - 0.6)
            ax.margins(y=0.15)
            if j == 2:
                ax.set_ylim(bottom=0)
            grid(ax, "y")
            ax.set_xlabel(xlabel if j == 0 else "")
            ax.set_title(ylabel)
            placers.append(end_labels(fig, ax, refs, t))
        row_axes[0].annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 26), textcoords="offset points",
                             fontsize=10.5, fontweight="bold", color=t["ink"], va="bottom")
    handles = [Line2D([], [], color=color, marker="o", ms=7, mec=t["surface"], label="LP optimizer run"),
               Line2D([], [], color=t["context"], lw=1, label="fixed policy (labeled at the right)"),
               Line2D([], [], ls="none", marker="_", ms=14, mew=1.6, color=t["ink"], label="the run's budget / ε")]
    if any(r["infeasible"] for _, _, sel, *_ in sweeps for r in sel):
        handles.append(Line2D([], [], ls="none", marker="o", ms=7, mfc=t["surface"], mec=color, mew=1.6,
                              label="hollow: no plan met every limit"))
    header(fig, t, "What the limits buy: the LP optimizer along a budget and an error sweep",
           "Every point is one training run; the plans are re-solved every k epochs from the same profile.", handles)
    finalize(fig, placers)
    return fig


def fig_loss(plt, ctx):
    ref = ctx.ref
    if ref is None:
        return None
    t = ctx.t
    others = [r for r in ctx.runs if r is not ref and r["losses"]]
    if not others:
        return None

    def rel(r):
        return [None if a is None or b is None else 100 * (a - b) / b for a, b in zip(r["losses"], ref["losses"])]
    curves = {r["name"]: rel(r) for r in others}
    vals = [v for c in curves.values() for v in c if v is not None]
    lim = max((abs(v) for v in vals), default=1) * 1.15 or 1
    n = len(others) + 1
    ncol = 4
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 0.9 + 2.15 * nrow), layout="constrained", squeeze=False)
    flat = axes.ravel()
    ax = flat[0]
    ep = list(range(len(ref["losses"])))
    ax.plot(ep, ref["losses"], color=ctx.color[ref["group"]])
    ax.set_title(f"{ref['name']}: training loss")
    grid(ax)
    for ax, r in zip(flat[1:], others):
        for other in others:
            if other is not r:
                c = curves[other["name"]]
                ax.plot(range(len(c)), c, color=t["context"], lw=0.9, zorder=1)
        c = curves[r["name"]]
        ax.axhline(0, color=t["baseline"], lw=0.8, zorder=1)
        ax.plot(range(len(c)), c, color=ctx.color[r["group"]], zorder=3)
        ax.set_ylim(-lim, lim)
        ax.set_title(label(r))
        grid(ax)
    for ax in flat[n:]:
        ax.set_visible(False)
    for ax in flat[:n]:
        ax.tick_params(labelsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    for i, row in enumerate(axes):
        row[0].set_ylabel("training loss" if i == 0 else "loss vs reference (%)")
    header(fig, t, f"Training loss of every run relative to {ref['name']}",
           "Per epoch, (loss − reference) / reference; the other runs in gray for context. "
           "Runs with the same fp16 layers repeat the same rounding, so their curves coincide.",
           group_handles(ctx, {r["group"] for r in ctx.runs}))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig


def policy_colors(t):
    return {p: tint(t["series"][hue], t["surface"]) if light else t["series"][hue] for p, _, hue, light in POLICIES}


def fig_plans(plt, ctx):
    runs = [r for r in ctx.runs if r["log"].get("replans")]
    if not runs:
        return None
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    t = ctx.t
    colors = policy_colors(t)
    ncol = 4
    nrow = -(-len(runs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 0.9 + 1.95 * nrow), layout="constrained", squeeze=False)
    flat = axes.ravel()
    g = 0.05
    for ax, r in zip(flat, runs):
        epochs = r["log"]["epochs"]
        nseg = max(len(e["policies"]) for e in epochs)
        for e in epochs:
            for s, p in enumerate(e["policies"]):
                ax.add_patch(Rectangle((e["epoch"] - 0.5 + g, s - 0.5 + g), 1 - 2 * g, 1 - 2 * g, lw=0,
                                       color=colors.get(p, t["context"])))
        for h in r["log"]["replans"]:
            ax.plot(h["epoch"], -0.85, marker="v", ms=5, color=t["secondary"] if h["feasible"] else t["muted"],
                    mfc=t["secondary"] if h["feasible"] else t["surface"], clip_on=False)
        ax.set_xlim(-0.5, len(epochs) - 0.5)
        ax.set_ylim(nseg - 0.5, -1.15)
        ax.set_yticks(range(nseg), [f"layer {s}" for s in range(nseg)])
        ax.set_xticks(range(len(epochs)))
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelsize=7.5)
        ax.annotate(run_mode(r), (0, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                    fontsize=7, color=t["muted"], va="bottom")
        ax.set_title(r["name"], pad=13)
    for ax in flat[len(runs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    handles = [Patch(color=colors[p], label=lbl) for p, lbl, _, _ in POLICIES]
    handles.append(Line2D([], [], ls="none", marker="v", ms=6, color=t["secondary"], label="re-plan (hollow: infeasible)"))
    header(fig, t, "The policy of every layer in every epoch",
           "Blue = fp32, orange = fp16 activations; the faded shade = recomputed in backward instead of stored. "
           "One panel per optimizer run.", handles)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig


def fig_replanning(plt, ctx):
    cands = [r for r in ctx.runs if any("noise" in h and h.get("segment_errors") for h in r["log"].get("replans", []))]
    if not cands:
        return None
    t = ctx.t
    run = max(cands, key=lambda r: (len(r["log"]["replans"]), r["name"]))
    hist = [h for h in run["log"]["replans"] if "noise" in h and h.get("segment_errors")]
    eps = run["log"]["config"].get("eps")
    metric = run["log"]["config"].get("metric", "grad")
    ep = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    neutral = t["secondary"]

    ax = axes[0]
    ax.plot(ep, [h["noise"] for h in hist], color=neutral, marker="o", ms=5, mec=t["surface"])
    ax.set_title("batch noise " + ("E‖g_b − ḡ‖²" if metric == "grad" else "Var(L_b)"))
    ax.set_ylim(bottom=0)
    grid(ax)

    ax = axes[1]
    nseg = len(hist[0]["segment_errors"])
    items = []
    for s in range(nseg):
        ys = [h["segment_errors"][s] for h in hist]
        ax.plot(ep, ys, color=t["series"][s % len(t["series"])], lw=1.8)
        items.append((ys[-1], f"layer {s}"))
    if eps is not None:
        ax.axhline(eps, color=t["ink"], lw=1, zorder=1)
        ax.annotate(f"ε = {eps:g}", (0, eps), xycoords=("axes fraction", "data"), xytext=(2, 3),
                    textcoords="offset points", fontsize=7.5, color=t["secondary"])
    ax.set_title("error of one layer in fp16 (RMS / noise)")
    ax.set_ylim(bottom=0, top=max(max(h["segment_errors"]) for h in hist + [{"segment_errors": [eps or 0]}]) * 1.2)
    grid(ax)
    placer = end_labels(fig, ax, items, t)

    ax = axes[2]
    pe = [(h["epoch"], h["predicted_error"]) for h in hist if h.get("predicted_error") is not None]
    if pe:
        ax.plot(*zip(*pe), color=neutral, marker="o", ms=5, mec=t["surface"])
    if eps is not None:
        ax.axhline(eps, color=t["ink"], lw=1, zorder=1)
        ax.annotate(f"ε = {eps:g}", (0, eps), xycoords=("axes fraction", "data"), xytext=(2, 3),
                    textcoords="offset points", fontsize=7.5, color=t["secondary"])
    ax.set_ylim(bottom=0, top=max([v for _, v in pe] + [eps or 0]) * 1.2)
    ax.set_title("predicted error of the chosen plan")
    grid(ax)
    for ax in axes:
        ax.set_xlabel("epoch of the re-plan")
    header(fig, t, f"What the re-plans measured in {run['name']} ({run_mode(run)})",
           "The errors are RMS fractions of the batch noise measured at the same re-plan; they add in squares, "
           "so ε admits roughly (ε / layer error)² layers in fp16.")
    finalize(fig, [placer])
    return fig


def fig_model(plt, ctx):
    pts = []
    for r in ctx.runs:
        plans = {}
        for e in r["log"]["epochs"]:
            if e.get("predicted_peak") is None or not e.get("steps"):
                continue
            plans.setdefault(tuple(e["policies"]), []).append(e)
        for es in plans.values():
            pts.append(dict(run=r, pred_peak=es[0]["predicted_peak"],
                            peak=max(e["measured_peak"] for e in es if e.get("measured_peak") is not None),
                            pred_ms=1e3 * es[0]["predicted_step_time"] if es[0].get("predicted_step_time") else None,
                            ms=statistics.median(e["seconds"] / e["steps"] * 1e3 for e in es)))
    if not pts:
        return None
    t = ctx.t
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    specs = [("peak memory", lambda p: p["pred_peak"] / GIB, lambda p: p["peak"] / GIB, "predicted peak (GiB)"),
             ("step time", lambda p: p["pred_ms"], lambda p: p["ms"], "predicted ms per step")]
    for ax, (what, pred, meas, xlabel) in zip(axes, specs):
        sel = [p for p in pts if pred(p) and meas(p)]
        if not sel:
            ax.set_visible(False)
            continue
        dev = [(pred(p), 100 * (meas(p) / pred(p) - 1), p["run"]) for p in sel]
        ax.axhline(0, color=t["baseline"], lw=0.8)
        scatter_runs(ax, ctx, dev)
        med = statistics.median(d for _, d, _ in dev)
        ax.annotate(f"median {med:+.2f}%  ·  range {min(d for _, d, _ in dev):+.2f}% to {max(d for _, d, _ in dev):+.2f}%",
                    (0, 1), xycoords="axes fraction", xytext=(4, -4), textcoords="offset points", va="top",
                    fontsize=8, color=t["secondary"])
        span = max(abs(d) for _, d, _ in dev) or 1
        ax.set_ylim(min(-0.25 * span, min(d for _, d, _ in dev) * 1.3), max(d for _, d, _ in dev) * 1.3 + 0.3 * span)
        ax.margins(x=0.08)
        grid(ax, "both")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("measured vs predicted (%)")
        ax.set_title(f"{what}: measured against the profiler's prediction")
    header(fig, t, "How well the cost model predicted every plan",
           "One dot per run and plan (epochs with the same policies pooled); above zero: more than predicted. Re-plans "
           "after the first add the memory measured but not modeled, hence the zeros; the step time also holds the "
           "optimizer step and data loading, which the profiler does not time.", group_handles(ctx, {p["run"]["group"] for p in pts}, any(p["run"]["infeasible"] for p in pts)))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig


BUILDERS = dict(tradeoff=fig_tradeoff, overview=fig_overview, sweeps=fig_sweeps, loss=fig_loss, plans=fig_plans,
                replanning=fig_replanning, model=fig_model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", default=["stats/experiments"], help="log files or folders of them")
    parser.add_argument("--reference", default="base_fp32", help="run to compare with (file name without .json)")
    parser.add_argument("--out", help="output folder [<first folder>/plots]")
    parser.add_argument("--only", nargs="+", choices=FIGURES, help="only these figures")
    parser.add_argument("--format", default="png", help="png, pdf or svg [png]")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    args = parser.parse_args(argv)

    logs = load(args.paths)
    if not logs:
        sys.exit("no training logs found")
    try:
        plt = setup(THEMES[args.theme])
    except ImportError:
        sys.exit("plot.py needs matplotlib: pip install matplotlib")
    runs, ref = collect(logs, args.reference)
    if ref is None:
        print(f"(no reference run {args.reference!r}: no relative numbers)")
    ctx = Context(runs, ref, args.theme)
    first = args.paths[0]
    out = args.out or os.path.join(first if os.path.isdir(first) else os.path.dirname(first) or ".", "plots")
    os.makedirs(out, exist_ok=True)
    written = []
    for name in args.only or FIGURES:
        fig = BUILDERS[name](plt, ctx)
        if fig is None:
            print(f"skip {name}: no runs with the data it needs")
            continue
        path = os.path.join(out, f"{name}.{args.format}")
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        written.append(path)
        print(path)
    return written


if __name__ == "__main__":
    main()
