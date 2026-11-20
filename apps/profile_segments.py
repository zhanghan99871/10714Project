"""Profile the segment policies of a Transformer LanguageModel and check the cost model.

    python apps/profile_segments.py                                  # cuda, "activations" size, tensor cores on
    python apps/profile_segments.py --size long --check 12 --out prof.json
    python apps/profile_segments.py --no-mma                         # fp16 matmuls without tensor cores
    python apps/profile_segments.py --device cpu --layers 3 --embedding 64 --hidden 256 --batch 8 --seq-len 32

Builds the model of apps/benchmark.py (random tokens, one fixed batch), runs
needle.policy.profile_segments and prints the cost table. It then measures complete
steps for the uniform configurations and `--check` random ones and compares them
with the predictions of the time and peak-memory models (the time model is fitted
on single-segment switches only, so every multi-segment configuration is unseen).
"""
import argparse
import sys

sys.path.append("./python")
sys.path.append(".")
import numpy as np

import needle as ndl
import needle.nn as nn
from apps.benchmark import SIZES, VOCAB
from apps.models import LanguageModel
from needle.policy import measure_step, profile_segments

MB = 2**20


def build(args):
    cfg = dict(SIZES[("transformer", args.size)])
    for name in ("num_layers", "embedding_size", "hidden_size", "batch", "seq_len"):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    device = {"cuda": ndl.cuda, "cpu": ndl.cpu, "cpu_numpy": ndl.cpu_numpy}[args.device]()
    if not device.enabled():
        sys.exit(f"the {args.device} backend is not available (no GPU?); try --device cpu")
    np.random.seed(args.seed)
    model = LanguageModel(cfg["embedding_size"], VOCAB, cfg["hidden_size"], num_layers=cfg["num_layers"],
                          seq_model="transformer", seq_len=cfg["seq_len"], device=device, policies="fp32")
    rng = np.random.RandomState(args.seed + 1)
    x = ndl.Tensor(rng.randint(0, VOCAB, (cfg["seq_len"], cfg["batch"])).astype(np.float32),
                   device=device, requires_grad=False)
    y = ndl.Tensor(rng.randint(0, VOCAB, cfg["seq_len"] * cfg["batch"]).astype(np.float32),
                   device=device, requires_grad=False)
    loss_fn = lambda: nn.SoftmaxLoss()(model(x)[0], y)
    return cfg, device, model, loss_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "cpu_numpy"])
    parser.add_argument("--size", default="activations", choices=sorted(s for m, s in SIZES if m == "transformer"))
    parser.add_argument("--layers", dest="num_layers", type=int)
    parser.add_argument("--embedding", dest="embedding_size", type=int)
    parser.add_argument("--hidden", dest="hidden_size", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq-len", dest="seq_len", type=int)
    parser.add_argument("--policies", default="fp32,fp16,none", help="policies to profile (must include fp32)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--check", type=int, default=8, help="random configurations to validate")
    parser.add_argument("--no-mma", action="store_true", help="profile with tensor cores off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="save the profile as JSON")
    args = parser.parse_args()

    cfg, device, model, loss_fn = build(args)
    policies = args.policies.split(",")
    mma = not args.no_mma
    key = f"{cfg['seq_len']}x{cfg['batch']}"
    print(f"config: {cfg}, device {args.device}, policies {policies}, mma={mma}")

    tr = model.sequence_model
    prof = profile_segments(tr, loss_fn, params=model.parameters(), policies=policies,
                            warmup=args.warmup, repeats=args.repeats, mma=mma, key=key)
    print(prof.summary())
    if args.out:
        prof.save(args.out)
        print(f"saved {args.out}")

    n = prof.num_segments
    rng = np.random.RandomState(args.seed + 2)
    configs = [[p] * n for p in policies] + [list(rng.choice(policies, n)) for _ in range(args.check)]
    print(f"\n{'policies':<{7 * n}} {'time pred':>10} {'measured':>10} {'err':>7}   "
          f"{'peak pred':>10} {'measured':>10} {'err':>7}")
    t_err, p_err = [], []
    for config in configs:
        r = measure_step(tr, loss_fn, config, params=model.parameters(), warmup=args.warmup,
                         repeats=args.repeats, mma=mma)
        tp, pp = prof.predict_time(config), prof.predict_peak(config, absolute=False)
        t_err.append(tp / r["time"] - 1)
        p_err.append(pp / r["peak"] - 1)
        print(f"{','.join(config):<{7 * n}} {tp * 1e3:8.2f}ms {r['time'] * 1e3:8.2f}ms {t_err[-1]:+7.1%}   "
              f"{pp / MB:8.1f}MB {r['peak'] / MB:8.1f}MB {p_err[-1]:+7.1%}")
    abs_t, abs_p = np.abs(t_err), np.abs(p_err)
    print(f"\ntime error: mean {abs_t.mean():.1%}, max {abs_t.max():.1%};  "
          f"peak error: mean {abs_p.mean():.2%}, max {abs_p.max():.2%} "
          f"(peaks are above the {prof.static / MB:.1f} MB held before a step)")


if __name__ == "__main__":
    main()
