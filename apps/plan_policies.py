"""Choose segment policies from a saved profile and error estimate (MILP, scipy >= 1.9).

    python apps/profile_segments.py --half-softmax --out stats/prof_act_hs.json    # time and memory
    python apps/estimate_errors.py --half-softmax --check 0 --out stats/err_act_hs.json
    python apps/plan_policies.py --budget-fraction 0.5 --eps 0.01      # reads the two files above
    python apps/plan_policies.py --frontier memory --eps 0.01
    python apps/plan_policies.py --frontier error --budget-mib 6000
    python apps/plan_policies.py --profile stats/prof_long.json --errors stats/err_long.json --budget-mib 8000

The profile and the error estimate default to stats/prof_act_hs.json and
stats/err_act_hs.json; without the default error file (or with --no-errors) the
plan has no error limit. See human-docs/policy-optimizer/user-guide.md.

The budget is the predicted peak of a whole step (static memory included), given
in MiB or as a fraction of the all-fp32 configuration's predicted peak; eps is the
allowed RMS precision error as a fraction of the batch noise (see
needle.policy.error). Prints the plan next to the uniform configurations, or a
Pareto frontier (time vs peak at --eps, or time vs error within the budget).
"""
import argparse
import os
import sys

sys.path.append("./python")
from needle.policy import SegmentErrors, SegmentProfile, error_frontier, memory_frontier, solve

MIB = 2 ** 20
DEFAULT_PROFILE = os.path.join("stats", "prof_act_hs.json")
DEFAULT_ERRORS = os.path.join("stats", "err_act_hs.json")


def table(plans, title):
    width = max([len(",".join(p.policies)) for p in plans] + [8]) + 1
    lines = [title, f"{'policies':<{width}} {'time ms':>9} {'peak MiB':>10} {'error':>10}"]
    for p in plans:
        err = "" if p.error is None else f"{p.error:10.3e}"
        lines.append(f"{','.join(p.policies):<{width}} {p.time * 1e3:9.2f} {p.peak / MIB:10.1f} {err}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=DEFAULT_PROFILE,
                        help=f"JSON saved by apps/profile_segments.py (default {DEFAULT_PROFILE})")
    parser.add_argument("--errors", default=None,
                        help=f"JSON saved by apps/estimate_errors.py (default {DEFAULT_ERRORS}, if it exists)")
    parser.add_argument("--no-errors", action="store_true", help="plan without the error estimate")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--budget-mib", type=float, help="memory budget of a step in MiB")
    budget.add_argument("--budget-fraction", type=float, help="budget as a fraction of the all-fp32 peak")
    parser.add_argument("--eps", type=float, help="allowed RMS error / batch noise")
    parser.add_argument("--metric", default="grad", choices=["grad", "loss"])
    parser.add_argument("--memory-margin", type=float, default=0.0, help="budget * (1 - margin)")
    parser.add_argument("--error-margin", type=float, default=1.0, help="eps / margin")
    parser.add_argument("--policies", help="comma-separated policies to choose from")
    parser.add_argument("--time-tolerance-ms", type=float, default=0.0,
                        help="prefer lower memory among plans this much slower than the fastest")
    parser.add_argument("--frontier", choices=["memory", "error"], help="print a Pareto frontier instead")
    args = parser.parse_args(argv)

    if not os.path.exists(args.profile):
        parser.error(f"no profile at {args.profile}: make one with apps/profile_segments.py --out "
                     f"{args.profile}, or pass --profile")
    profile = SegmentProfile.load(args.profile)
    errors_path = None
    if not args.no_errors:
        if args.errors is not None:
            errors_path = args.errors  # given explicitly: it must exist
        elif os.path.exists(DEFAULT_ERRORS):
            errors_path = DEFAULT_ERRORS
    errors = SegmentErrors.load(errors_path) if errors_path else None
    policies = args.policies.split(",") if args.policies else None
    if args.budget_fraction is not None:
        budget = args.budget_fraction * profile.predict_peak("fp32")
    elif args.budget_mib is not None:
        budget = args.budget_mib * MIB
    else:
        budget = None
    print(f"profile {args.profile}: {profile.num_segments} segments on {profile.device} (key {profile.key}, "
          f"half_softmax={profile.half_softmax}); all-fp32 step {profile.base_time * 1e3:.1f} ms, "
          f"peak {profile.predict_peak('fp32') / MIB:.1f} MiB")
    print(f"errors: {errors_path}" if errors_path else "errors: none (no error limit)")
    if args.eps is not None and errors is None:
        parser.error("--eps needs an error estimate (--errors FILE)")

    if args.frontier == "memory":
        plans = memory_frontier(profile, errors, eps=args.eps, metric=args.metric, policies=policies)
        print(table(plans, f"\ntime / peak Pareto frontier (eps {args.eps}, {args.metric} error):"))
        return plans
    if args.frontier == "error":
        if errors is None:
            parser.error("--frontier error needs --errors")
        plans = error_frontier(profile, errors, budget=budget, metric=args.metric, policies=policies)
        title = "no budget" if budget is None else f"budget {budget / MIB:.1f} MiB"
        print(table(plans, f"\ntime / {args.metric} error Pareto frontier ({title}):"))
        return plans

    plan = solve(profile, errors, budget=budget, eps=args.eps, metric=args.metric,
                 memory_margin=args.memory_margin, error_margin=args.error_margin, policies=policies,
                 time_tolerance=args.time_tolerance_ms / 1e3)
    print("\n" + plan.summary())
    uniform = []
    for p in (policies or profile.policies):
        u = solve(profile, errors, policies=[p], metric=args.metric)  # the only configuration
        uniform.append(u)
    print(table(uniform, "\nuniform configurations for comparison:"))
    return plan


if __name__ == "__main__":
    main()
