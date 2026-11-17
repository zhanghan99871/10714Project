"""VRAM and speed of training steps: automatic mixed precision and gradient checkpointing.

    python apps/benchmark.py --model transformer --size activations --amp float16 --checkpoint
    python apps/benchmark.py --suite amp --out amp.json          # every AMP configuration
    python apps/benchmark.py --suite checkpoint --out ckpt.json  # every checkpoint configuration

Each configuration runs in its own process, so runs do not share memory. A run trains
a LanguageModel on random tokens with apps/simple_ml.epoch_general_ptb: `--warmup`
untimed steps, then `--steps` timed steps with the memory counter on (peak memory,
time, loss), then `--steps` more with the counter off (time only, to show the
counter's own cost). Memory is needle's buffer counter
(needle.backend_ndarray.max_memory_allocated); it covers NDArray buffers only.
"""
import argparse
import contextlib
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
import needle.nn.nn_transformer as nn_transformer
import apps.simple_ml as simple_ml
from apps.models import LanguageModel
from apps.simple_ml import epoch_general_ptb

# The vocabulary stays small in every size: Embedding and SoftmaxLoss build one-hot
# rows from np.eye(vocab) on the host at every step.
SIZES = {
    ("transformer", "activations"): dict(embedding_size=256, hidden_size=1024, num_layers=4, batch=32, seq_len=256),
    ("transformer", "weights"): dict(embedding_size=1024, hidden_size=4096, num_layers=4, batch=4, seq_len=16),
    ("transformer", "long"): dict(embedding_size=256, hidden_size=1024, num_layers=4, batch=32, seq_len=512),
    ("lstm", "activations"): dict(embedding_size=256, hidden_size=512, num_layers=2, batch=64, seq_len=64),
    ("lstm", "weights"): dict(embedding_size=1024, hidden_size=2048, num_layers=2, batch=4, seq_len=8),
}
VOCAB = 1000
AMPS = ("float32", "float16", "bfloat16")
MB = 2**20

# (model, size, amp, checkpoint, release_inputs)
SUITES = {
    "amp": [(m, s, a, False, False) for m, s in [("transformer", "activations"), ("transformer", "weights"),
                                                ("lstm", "activations"), ("lstm", "weights")] for a in AMPS],
    "checkpoint": [("transformer", s, a, c, False) for s in ("activations", "long") for a in AMPS for c in (False, True)]
    + [("transformer", "activations", "float16", c, True) for c in (False, True)],
}


def autocast_context(amp, release_inputs):
    if amp == "float32":
        return contextlib.nullcontext()
    return ndl.autocast(dtype=amp, release_inputs=release_inputs)


def released_segment_inputs(model, cfg, device, amp, release_inputs):
    """Forward once and count the segment inputs (non-leaf) that a cast released."""
    inputs, segment = [], nn_transformer.checkpoint
    nn_transformer.checkpoint = lambda fn, x: (inputs.append(x), segment(fn, x))[1]
    X = ndl.Tensor(np.random.randint(0, VOCAB, size=(cfg["seq_len"], cfg["batch"])).astype(np.float32),
                   device=device, requires_grad=False)
    with autocast_context(amp, release_inputs):
        model(X)
    nn_transformer.checkpoint = segment
    return sum(1 for x in inputs if not x.is_leaf() and x.cached_data is None)


def run(model_name, size, amp, steps, warmup, checkpoint=False, release_inputs=False):
    cfg = SIZES[(model_name, size)]
    device = ndl.cuda()
    np.random.seed(0)
    model = LanguageModel(cfg["embedding_size"], VOCAB, cfg["hidden_size"], num_layers=cfg["num_layers"],
                          seq_model=model_name, seq_len=cfg["seq_len"], device=device, checkpoint=checkpoint)
    params = sum(int(np.prod(p.shape)) for p in model.parameters())
    opt = ndl.optim.SGD(model.parameters(), lr=0.01)  # 0.1 diverges on random tokens
    scaler = ndl.amp.GradScaler(enabled=amp == "float16")
    # epoch_general_ptb opens ndl.autocast(dtype) itself; this adds release_inputs
    simple_ml._amp_context = lambda amp_arg: autocast_context(amp, release_inputs)

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
    released = released_segment_inputs(model, cfg, device, amp, release_inputs) if checkpoint else None
    ndarray_module._track_allocation = lambda array: None
    step_time_no_counter, _ = epoch(steps)
    return dict(model=model_name, size=size, amp=amp, checkpoint=checkpoint, release_inputs=release_inputs,
                **cfg, vocab=VOCAB, params=params, steps=steps, base_mb=base / MB, peak_mb=peak / MB,
                ms_per_step=step_time * 1e3, ms_per_step_no_counter=step_time_no_counter * 1e3, loss=loss,
                skipped_steps=skipped, released_segment_inputs=released)


def run_suite(suite, steps, warmup, out):
    results = []
    for model_name, size, amp, checkpoint, release_inputs in SUITES[suite]:
        cmd = [sys.executable, __file__, "--model", model_name, "--size", size, "--amp", amp,
               "--steps", str(steps), "--warmup", str(warmup), "--json"]
        cmd += ["--checkpoint"] * checkpoint + ["--release-inputs"] * release_inputs
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            results.append(dict(model=model_name, size=size, amp=amp, checkpoint=checkpoint,
                                release_inputs=release_inputs, error=proc.stderr.strip().splitlines()[-1]))
        else:
            results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        print(json.dumps(results[-1]), flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    print_table(results)


def print_table(results):
    """Ratios compare each run with float32 without checkpoint of the same model and size."""
    print("| model | size | amp | ckpt | release_inputs | peak MB | peak ratio | ms/step | speedup | "
          "ms/step, no counter | loss | skipped | released inputs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    ref = {(r["model"], r["size"]): r for r in results
           if r["amp"] == "float32" and not r.get("checkpoint") and "error" not in r}
    for r in results:
        head = f"| {r['model']} | {r['size']} | {r['amp']} | {r.get('checkpoint', False)} | {r.get('release_inputs', False)} |"
        if "error" in r:
            print(f"{head} error: {r['error']} |")
            continue
        base = ref.get((r["model"], r["size"]))
        mem = f"{r['peak_mb'] / base['peak_mb']:.2f}" if base else "-"
        speed = f"{base['ms_per_step'] / r['ms_per_step']:.2f}" if base else "-"
        released = r.get("released_segment_inputs")
        print(f"{head} {r['peak_mb']:.0f} | {mem} | {r['ms_per_step']:.0f} | {speed} | "
              f"{r['ms_per_step_no_counter']:.0f} | {r['loss']:.4f} | {r['skipped_steps']} | "
              f"{'-' if released is None else released} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("transformer", "lstm"), default="transformer")
    parser.add_argument("--size", choices=("activations", "weights", "long"), default="activations")
    parser.add_argument("--amp", choices=AMPS, default="float32")
    parser.add_argument("--checkpoint", action="store_true", help="one checkpointed segment per transformer layer")
    parser.add_argument("--release-inputs", action="store_true", help="autocast(release_inputs=True)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--suite", choices=SUITES, help="run every configuration of a suite, one process each")
    parser.add_argument("--out", help="with --suite: write the results as JSON to this file")
    parser.add_argument("--json", action="store_true", help="print the result as one JSON line")
    args = parser.parse_args()
    if args.suite:
        run_suite(args.suite, args.steps, args.warmup, args.out)
    else:
        result = run(args.model, args.size, args.amp, args.steps, args.warmup, args.checkpoint, args.release_inputs)
        print(json.dumps(result) if args.json else result)
