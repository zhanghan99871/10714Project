"""Train a Transformer LanguageModel with dynamically planned segment policies.

    python apps/train_policies.py --data data/ptb --budget-fraction 0.4 --eps 0.01 --replan-every 5 \\
        --epochs 20 --half-softmax --log stats/train_ptb_b40_e01.json
    python apps/train_policies.py --data data/ptb --static fp16 --epochs 20 --half-softmax \\
        --log stats/train_ptb_fp16.json                                   # a baseline, same log format
    python apps/train_policies.py --data random --device cpu --layers 3 --embedding 64 --hidden 256 \\
        --batch 4 --seq-len 32 --epochs 3 --budget-fraction 0.5 --eps 0.01   # quick check, no data needed

Builds the model of apps/benchmark.py (or your sizes) on Penn Treebank (a folder
with train.txt and test.txt, see needle.data.datasets.Corpus) or on random tokens,
and trains it with apps/simple_ml.train_ptb and a needle.policy.PolicyController:
the policies are planned before epoch 0 and every --replan-every epochs, from the
memory budget and the error limit eps. --static trains with fixed policies instead
(nothing is profiled or planned) and logs the same way, for baselines. --profile
uses a saved profile instead of profiling at the start; it must come from the same
model (sizes and vocabulary), batch shape and --half-softmax: stats/prof_act_hs.json
(random tokens, vocabulary 1000) fits --data random, not PTB.

The log (JSON) has the controller configuration, every re-plan (policies,
predictions, the estimated errors, the time spent) and every epoch (loss,
accuracy, time, measured and predicted peak, loss scale), and the final test
loss and perplexity. See human-docs/policy-optimizer/user-guide.md.
"""
import argparse
import math
import sys
import time

sys.path.append("./python")
sys.path.append(".")
import numpy as np

import needle as ndl
from apps.benchmark import SIZES, VOCAB
from apps.models import LanguageModel
from apps.simple_ml import evaluate_ptb, train_ptb
from needle.policy import PolicyController, SegmentProfile

MIB = 2 ** 20


def load_data(args, cfg):
    """(train data, test data or None, vocabulary size), batchified."""
    if args.data == "random":
        rng = np.random.RandomState(args.seed)
        rows = cfg["seq_len"] * args.random_steps + 1
        train = rng.randint(0, VOCAB, (rows, cfg["batch"])).astype(np.float32)
        return train, None, VOCAB
    corpus = ndl.data.Corpus(args.data, max_lines=args.max_lines)
    train = ndl.data.batchify(corpus.train, cfg["batch"], device=None, dtype="float32")
    test = ndl.data.batchify(corpus.test, cfg["batch"], device=None, dtype="float32")
    return train, test, len(corpus.dictionary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/ptb", help="PTB folder (train.txt, test.txt) or 'random'")
    parser.add_argument("--max-lines", type=int, help="read at most this many lines of each PTB file")
    parser.add_argument("--random-steps", type=int, default=20, help="steps per epoch with --data random")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "cpu_numpy"])
    parser.add_argument("--size", default="activations", choices=sorted(s for m, s in SIZES if m == "transformer"))
    parser.add_argument("--layers", dest="num_layers", type=int)
    parser.add_argument("--embedding", dest="embedding_size", type=int)
    parser.add_argument("--hidden", dest="hidden_size", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq-len", dest="seq_len", type=int)
    parser.add_argument("--half-softmax", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, help="clip the gradient norm")
    parser.add_argument("--seed", type=int, default=0)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--budget-fraction", type=float, help="budget as a fraction of the all-fp32 peak")
    budget.add_argument("--budget-mib", type=float, help="memory budget of a step in MiB")
    parser.add_argument("--eps", type=float, help="allowed RMS error / batch noise")
    parser.add_argument("--metric", default="grad", choices=["grad", "loss"])
    parser.add_argument("--replan-every", type=int, default=5, help="re-plan every k epochs")
    parser.add_argument("--error-batches", type=int, default=8)
    parser.add_argument("--grad-batches", type=int, default=2)
    parser.add_argument("--memory-margin", type=float, default=0.0)
    parser.add_argument("--error-margin", type=float, default=1.0)
    parser.add_argument("--policies", help="comma-separated policies to choose from")
    parser.add_argument("--profile", help="use this saved profile instead of profiling")
    parser.add_argument("--static", help="train with fixed policies (one, or one per segment, comma-separated)")
    parser.add_argument("--log", help="write the training log (JSON) here, e.g. stats/train_run.json")
    parser.add_argument("--no-eval", action="store_true", help="skip the final test evaluation")
    parser.add_argument("--verbose", action="store_true", help="enable verbose output")
    args = parser.parse_args(argv)

    cfg = dict(SIZES[("transformer", args.size)])
    for name in ("num_layers", "embedding_size", "hidden_size", "batch", "seq_len"):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    device = {"cuda": ndl.cuda, "cpu": ndl.cpu, "cpu_numpy": ndl.cpu_numpy}[args.device]()
    if not device.enabled():
        sys.exit(f"the {args.device} backend is not available (no GPU?); try --device cpu")
    train, test, vocab = load_data(args, cfg)
    np.random.seed(args.seed)
    model = LanguageModel(cfg["embedding_size"], vocab, cfg["hidden_size"], num_layers=cfg["num_layers"],
                          seq_model="transformer", seq_len=cfg["seq_len"], device=device, policies="fp32",
                          half_softmax=args.half_softmax)
    budget = None if args.budget_mib is None else args.budget_mib * MIB
    static = None
    if args.static:
        static = args.static.split(",")
        static = static[0] if len(static) == 1 else static
    controller = PolicyController(
        model, budget=budget, budget_fraction=args.budget_fraction, eps=args.eps, metric=args.metric,
        replan_every=args.replan_every, error_batches=args.error_batches, grad_batches=args.grad_batches,
        memory_margin=args.memory_margin, error_margin=args.error_margin,
        policies=args.policies.split(",") if args.policies else None,
        profile=SegmentProfile.load(args.profile) if args.profile else None,
        static_policies=static, log_path=args.log)
    controller.results["config"] = {**cfg, "vocab": vocab, "data": args.data, "device": args.device,
                                    "optimizer": args.optimizer, "lr": args.lr, "epochs": args.epochs}
    print(f"model {cfg}, vocab {vocab}, {train.shape[0] // cfg['seq_len']} steps per epoch, device {args.device}")

    optimizer = {"adam": ndl.optim.Adam, "sgd": ndl.optim.SGD}[args.optimizer]
    start = time.perf_counter()
    acc, loss = train_ptb(model, train, seq_len=cfg["seq_len"], n_epochs=args.epochs, optimizer=optimizer,
                          lr=args.lr, clip=args.clip, device=device, controller=controller, verbose=args.verbose)
    controller.results["train_seconds"] = time.perf_counter() - start
    controller.results["train_loss"], controller.results["train_acc"] = float(loss), float(acc)

    if controller.history:
        print(controller.summary())
    print(f"\n{'epoch':>5} {'policies':<42} {'loss':>8} {'time s':>8} {'peak MiB':>9} {'pred MiB':>9}")
    for e in controller.epochs:
        pred = "" if e["predicted_peak"] is None else f"{e['predicted_peak'] / MIB:9.1f}"
        print(f"{e['epoch']:5d} {','.join(e['policies']):<42} {e['loss']:8.4f} {e['seconds']:8.1f} "
              f"{e['measured_peak'] / MIB:9.1f} {pred}")
    if test is not None and not args.no_eval:
        test_acc, test_loss = evaluate_ptb(model, test, seq_len=cfg["seq_len"], device=device)
        controller.results.update(test_loss=float(test_loss), test_acc=float(test_acc),
                                  test_perplexity=float(math.exp(test_loss)))
        print(f"test loss {test_loss:.4f}, perplexity {math.exp(test_loss):.2f}")
    controller.save_log()
    if args.log:
        print(f"log: {args.log}")
    return controller


if __name__ == "__main__":
    main()
