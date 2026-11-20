"""Estimate the per-segment float16 error of a Transformer LanguageModel and check it.

    python apps/estimate_errors.py                                  # cuda, "activations" size
    python apps/estimate_errors.py --half-softmax --out err.json
    python apps/estimate_errors.py --device cpu --layers 3 --embedding 64 --hidden 256 --batch 4 --seq-len 64

Builds the model of apps/benchmark.py on random tokens, runs
needle.policy.estimate_segment_errors over --batches batches and prints the error
table. It then measures, on the same batches, the actual loss error and relative
gradient error of the uniform configurations and --check random ones, and compares
them with the first-order predictions (predict(): the linear bound the optimizer
uses; predict_signed(): |sum_i s_i|).
"""
import argparse
import sys
import time

sys.path.append("./python")
sys.path.append(".")
import numpy as np

import needle as ndl
import needle.nn as nn
from apps.benchmark import SIZES, VOCAB
from apps.models import LanguageModel
from needle.policy import estimate_segment_errors, measure_precision_error


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "cpu_numpy"])
    parser.add_argument("--size", default="activations", choices=sorted(s for m, s in SIZES if m == "transformer"))
    parser.add_argument("--layers", dest="num_layers", type=int)
    parser.add_argument("--embedding", dest="embedding_size", type=int)
    parser.add_argument("--hidden", dest="hidden_size", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq-len", dest="seq_len", type=int)
    parser.add_argument("--batches", type=int, default=4, help="batches the estimate averages over")
    parser.add_argument("--check", type=int, default=12, help="random configurations to validate")
    parser.add_argument("--half-softmax", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="save the estimate as JSON")
    args = parser.parse_args()

    cfg = dict(SIZES[("transformer", args.size)])
    for name in ("num_layers", "embedding_size", "hidden_size", "batch", "seq_len"):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    device = {"cuda": ndl.cuda, "cpu": ndl.cpu, "cpu_numpy": ndl.cpu_numpy}[args.device]()
    if not device.enabled():
        sys.exit(f"the {args.device} backend is not available (no GPU?); try --device cpu")
    np.random.seed(args.seed)
    model = LanguageModel(cfg["embedding_size"], VOCAB, cfg["hidden_size"], num_layers=cfg["num_layers"],
                          seq_model="transformer", seq_len=cfg["seq_len"], device=device, policies="fp32",
                          half_softmax=args.half_softmax)
    fns = []
    for b in range(args.batches):
        rng = np.random.RandomState(args.seed + 100 + b)
        x = ndl.Tensor(rng.randint(0, VOCAB, (cfg["seq_len"], cfg["batch"])).astype(np.float32),
                       device=device, requires_grad=False)
        y = ndl.Tensor(rng.randint(0, VOCAB, cfg["seq_len"] * cfg["batch"]).astype(np.float32),
                       device=device, requires_grad=False)
        fns.append(lambda x=x, y=y: nn.SoftmaxLoss()(model(x)[0], y))
    print(f"config: {cfg}, device {args.device}, half_softmax={args.half_softmax}, {args.batches} batches")

    tr = model.sequence_model
    start = time.perf_counter()
    est = estimate_segment_errors(tr, fns, params=model.parameters())
    print(f"estimate: {time.perf_counter() - start:.2f} s for {args.batches} batches")
    print(est.summary())
    if args.out:
        est.save(args.out)
        print(f"saved {args.out}")

    n = est.num_segments
    rng = np.random.RandomState(args.seed + 2)
    policies = est.policies
    configs = [[p] * n for p in policies] + [list(rng.choice(policies, n)) for _ in range(args.check)]
    width = max(len(",".join(c)) for c in configs) + 1
    print(f"\n{'policies':<{width}} {'bound':>9} {'1st-order':>9} {'measured':>9} {'grad err':>9}")
    rows = []
    for config in configs:
        r = measure_precision_error(tr, fns, config, params=model.parameters())
        bound, signed = est.predict(config), est.predict_signed(config)
        rows.append((bound, signed, r["loss_error"], r["grad_error"]))
        print(f"{','.join(config):<{width}} {bound:9.2e} {signed:9.2e} {r['loss_error']:9.2e} {r['grad_error']:9.2e}")
    half = [r for r in rows if r[0] > 0]
    if len(half) > 2:
        b, s, a, g = (np.array(v) for v in zip(*half))
        print(f"\nover the {len(half)} configurations with float16 segments: rank correlation of the bound "
              f"with the measured loss error {rank_corr(b, a):.2f}, with the gradient error {rank_corr(b, g):.2f}; "
              f"measured <= bound for {np.mean(a <= b):.0%} (largest measured / bound {np.max(a / b):.2f}); "
              f"first-order |sum s_i| vs measured: median ratio {np.median(s / a):.2f}")


if __name__ == "__main__":
    main()
