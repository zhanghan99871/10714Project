"""VRAM and speed of training steps with automatic mixed precision.

    python apps/benchmark.py --model transformer --size activations --amp float16
    python apps/benchmark.py --all --out results.json   # every configuration

Each configuration runs in its own process, so runs do not share memory. A run trains
a LanguageModel on random tokens with apps/simple_ml.epoch_general_ptb: `--warmup`
untimed steps, then `--steps` timed steps with the memory counter on (peak memory,
time, loss), then `--steps` more with the counter off (time only, to show the
counter's own cost). Memory is needle's buffer counter
(needle.backend_ndarray.max_memory_allocated); it covers NDArray buffers only.
"""
import argparse
import json
import subprocess
import sys
import time

sys.path.append("./python")
sys.path.append(".")
import numpy as np

import needle as ndl
import needle.backend_ndarray as nd
import needle.backend_ndarray.ndarray as ndarray_module
from apps.models import LanguageModel
from apps.simple_ml import epoch_general_ptb

# The vocabulary stays small in every size: Embedding and SoftmaxLoss build one-hot
# rows from np.eye(vocab) on the host at every step.
SIZES = {
    ("transformer", "activations"): dict(embedding_size=256, hidden_size=1024, num_layers=4, batch=32, seq_len=256),
    ("transformer", "weights"): dict(embedding_size=1024, hidden_size=4096, num_layers=4, batch=4, seq_len=16),
    ("lstm", "activations"): dict(embedding_size=256, hidden_size=512, num_layers=2, batch=64, seq_len=64),
    ("lstm", "weights"): dict(embedding_size=1024, hidden_size=2048, num_layers=2, batch=4, seq_len=8),
}
VOCAB = 1000
AMPS = ("float32", "float16", "bfloat16")
MB = 2**20


def run(model_name, size, amp, steps, warmup):
    cfg = SIZES[(model_name, size)]
    device = ndl.cuda()
    np.random.seed(0)
    model = LanguageModel(cfg["embedding_size"], VOCAB, cfg["hidden_size"], num_layers=cfg["num_layers"],
                          seq_model=model_name, seq_len=cfg["seq_len"], device=device)
    params = sum(int(np.prod(p.shape)) for p in model.parameters())
    opt = ndl.optim.SGD(model.parameters(), lr=0.01)  # 0.1 diverges on random tokens
    scaler = ndl.amp.GradScaler(enabled=amp == "float16")

    def epoch(n):
        data = np.random.randint(0, VOCAB, size=(n * cfg["seq_len"] + 1, cfg["batch"])).astype(np.float32)
        start = time.perf_counter()
        _, loss = epoch_general_ptb(data, model, seq_len=cfg["seq_len"], opt=opt, device=device,
                                    amp=False if amp == "float32" else amp, scaler=scaler)
        return (time.perf_counter() - start) / n, loss

    epoch(warmup)
    base = nd.memory_allocated(device)
    nd.reset_peak_memory_stats(device)
    skipped = scaler.skipped_steps
    step_time, loss = epoch(steps)
    peak = nd.max_memory_allocated(device)
    skipped = scaler.skipped_steps - skipped
    ndarray_module._track_allocation = lambda array: None
    step_time_no_counter, _ = epoch(steps)
    return dict(model=model_name, size=size, amp=amp, **cfg, vocab=VOCAB, params=params, steps=steps,
                base_mb=base / MB, peak_mb=peak / MB, ms_per_step=step_time * 1e3,
                ms_per_step_no_counter=step_time_no_counter * 1e3, loss=loss, skipped_steps=skipped)


def run_all(steps, warmup, out):
    results = []
    for model_name, size in SIZES:
        for amp in AMPS:
            cmd = [sys.executable, __file__, "--model", model_name, "--size", size, "--amp", amp,
                   "--steps", str(steps), "--warmup", str(warmup), "--json"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                results.append(dict(model=model_name, size=size, amp=amp, error=proc.stderr.strip().splitlines()[-1]))
            else:
                results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
            print(json.dumps(results[-1]), flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    print_table(results)


def print_table(results):
    print("| model | size | amp | peak MB | peak vs fp32 | ms/step | speedup vs fp32 | ms/step, no counter | loss | skipped |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    fp32 = {(r["model"], r["size"]): r for r in results if r["amp"] == "float32" and "error" not in r}
    for r in results:
        if "error" in r:
            print(f"| {r['model']} | {r['size']} | {r['amp']} | error: {r['error']} |")
            continue
        ref = fp32.get((r["model"], r["size"]))
        mem = f"{r['peak_mb'] / ref['peak_mb']:.2f}" if ref else "-"
        speed = f"{ref['ms_per_step'] / r['ms_per_step']:.2f}" if ref else "-"
        print(f"| {r['model']} | {r['size']} | {r['amp']} | {r['peak_mb']:.0f} | {mem} | {r['ms_per_step']:.0f} | "
              f"{speed} | {r['ms_per_step_no_counter']:.0f} | {r['loss']:.4f} | {r['skipped_steps']} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("transformer", "lstm"), default="transformer")
    parser.add_argument("--size", choices=("activations", "weights"), default="activations")
    parser.add_argument("--amp", choices=AMPS, default="float32")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--all", action="store_true", help="run every configuration, one process each")
    parser.add_argument("--out", help="with --all: write the results as JSON to this file")
    parser.add_argument("--json", action="store_true", help="print the result as one JSON line")
    args = parser.parse_args()
    if args.all:
        run_all(args.steps, args.warmup, args.out)
    else:
        result = run(args.model, args.size, args.amp, args.steps, args.warmup)
        print(json.dumps(result) if args.json else result)
