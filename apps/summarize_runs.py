"""Compare training runs of apps/train_policies.py: time, memory and error.

    python apps/summarize_runs.py stats/experiments --reference base_fp32 --csv stats/experiments/summary.csv
    python apps/summarize_runs.py stats/train_ptb_fp32.json stats/train_ptb_b40_e01.json --reference train_ptb_fp32

For every run (one JSON log each):

    mode         what chose the policies: fixed, or the LP optimizer (budget, eps, re-plan
                 period, metric, allowed policies, half_softmax)
    policies     the policies of the last epoch; plans: how many different configurations
                 the run trained with
    ms/step      median over the epochs; speed-up: reference ms/step / ms/step
    peak MiB     needle's memory counter, the largest of the epochs; saved: vs the reference
    loss         training loss of the last epoch; d loss: relative difference to the
                 reference at the last epoch; max |d loss|: the largest over the epochs
    test ppl     test perplexity; d ppl: relative difference to the reference
    pred. error  the predicted RMS precision error of the last plan (fraction of the batch noise)
    planning s   time spent profiling, estimating the errors and solving
    notes        infeasible plans, epochs over the budget, optimizer steps skipped by the
                 loss scaler
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

MIB = 2 ** 20


def load(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    runs = {}
    for f in files:
        with open(f) as fh:
            try:
                log = json.load(fh)
            except json.JSONDecodeError:
                continue
        if not isinstance(log, dict) or "epochs" not in log or "config" not in log:
            continue  # e.g. a saved profile
        runs[os.path.splitext(os.path.basename(f))[0]] = log
    return runs


def mode(log):
    c = log["config"]
    if c.get("static_policies"):
        pol = c["static_policies"]
        return f"fixed {pol[0]}" if len(set(pol)) == 1 else f"fixed {','.join(pol)}"
    parts = ["LP"]
    if c.get("budget_fraction") is not None:
        parts.append(f"b={c['budget_fraction']:g}")
    elif c.get("budget") is not None:
        parts.append(f"b={c['budget'] / MIB:.0f}MiB")
    if c.get("eps") is not None:
        parts.append(f"eps={c['eps']:g}" + ("(loss)" if c.get("metric") == "loss" else ""))
    parts.append(f"k={c['replan_every']}")
    if c.get("policies"):
        parts.append("{" + ",".join(c["policies"]) + "}")
    if c.get("grad_batches") not in (None, 2):
        parts.append(f"gb={c['grad_batches']}")
    if c.get("half_softmax") is False:
        parts.append("no-hs")
    return " ".join(parts)


def stats(log):
    epochs = log["epochs"]
    ms = [e["seconds"] / e["steps"] * 1e3 for e in epochs if e.get("seconds") and e.get("steps")]
    peaks = [e["measured_peak"] for e in epochs if e.get("measured_peak") is not None]
    configs = []
    for e in epochs:
        if e["policies"] not in configs:
            configs.append(e["policies"])
    replans = log.get("replans", [])
    notes = []
    infeasible = sum(not h["feasible"] for h in replans)
    if infeasible:
        notes.append(f"infeasible {infeasible}/{len(replans)}")
    over = sum(bool(e.get("over_budget")) for e in epochs)
    if over:
        notes.append(f"over budget {over} ep")
    skipped = epochs[-1].get("skipped_steps") if epochs else 0
    if skipped:
        notes.append(f"skipped {skipped} steps")
    return {
        "mode": mode(log),
        "policies": ",".join(epochs[-1]["policies"]) if epochs else "",
        "plans": len(configs),
        "replans": len(replans),
        "epochs": len(epochs),
        "ms_step": statistics.median(ms) if ms else None,
        "peak_mib": max(peaks) / MIB if peaks else None,
        "loss": epochs[-1].get("loss") if epochs else None,
        "losses": [e.get("loss") for e in epochs],
        "test_ppl": log.get("results", {}).get("test_perplexity"),
        "pred_error": replans[-1]["predicted_error"] if replans else None,
        "planning_s": sum(sum(h["seconds"].values()) for h in replans),
        "notes": "; ".join(notes),
    }


def compare(row, ref):
    def rel(a, b):
        return None if a is None or b in (None, 0) else (a - b) / b

    row["speedup"] = None if not ref or not row["ms_step"] or not ref["ms_step"] else ref["ms_step"] / row["ms_step"]
    row["mem_saved"] = None if not ref or row["peak_mib"] is None or not ref["peak_mib"] else 1 - row["peak_mib"] / ref["peak_mib"]
    row["d_loss"] = rel(row["loss"], ref["loss"]) if ref else None
    pairs = [(a, b) for a, b in zip(row["losses"], ref["losses"]) if a is not None and b is not None] if ref else []
    row["max_d_loss"] = max(abs(a - b) / abs(b) for a, b in pairs) if pairs else None
    row["d_ppl"] = rel(row["test_ppl"], ref["test_ppl"]) if ref else None
    return row


def fmt(v, spec, empty=""):
    return empty if v is None else format(v, spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="log files or folders of them")
    parser.add_argument("--reference", default="base_fp32", help="run to compare with (file name without .json)")
    parser.add_argument("--csv", help="also write the table as CSV")
    args = parser.parse_args(argv)

    runs = load(args.paths)
    if not runs:
        sys.exit("no training logs found")
    rows = {name: stats(log) for name, log in runs.items()}
    ref = rows.get(args.reference)
    if ref is None:
        print(f"(no reference run {args.reference!r}: no relative columns)")
    for name, row in rows.items():
        compare(row, ref)
        row["run"] = name

    head = ["run", "mode", "policies (last epoch)", "plans", "ms/step", "speed-up", "peak MiB", "saved",
            "loss", "d loss", "max |d loss|", "test ppl", "d ppl", "pred. error", "planning s", "notes"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name in sorted(rows, key=lambda n: (n != args.reference, n)):
        r = rows[name]
        cells = [name, r["mode"], r["policies"], str(r["plans"]), fmt(r["ms_step"], ".0f"), fmt(r["speedup"], ".2f") + ("x" if r["speedup"] else ""),
                 fmt(r["peak_mib"], ".0f"), fmt(r["mem_saved"], "+.0%"), fmt(r["loss"], ".4f"), fmt(r["d_loss"], "+.2%"),
                 fmt(r["max_d_loss"], ".2%"), fmt(r["test_ppl"], ".1f"), fmt(r["d_ppl"], "+.2%"),
                 fmt(r["pred_error"], ".2e"), fmt(r["planning_s"], ".0f") if r["replans"] else "", r["notes"]]
        print("| " + " | ".join(cells) + " |")

    if args.csv:
        keys = ["run", "mode", "policies", "plans", "replans", "epochs", "ms_step", "speedup", "peak_mib", "mem_saved",
                "loss", "d_loss", "max_d_loss", "test_ppl", "d_ppl", "pred_error", "planning_s", "notes"]
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for name in sorted(rows):
                w.writerow(rows[name])
        print(f"\nCSV: {args.csv}")
    return rows


if __name__ == "__main__":
    main()
