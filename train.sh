#!/usr/bin/env bash
# Experiments of the dynamic mixed-precision / checkpoint policy optimizer on Penn Treebank.
#
#   ./train.sh baseline     fixed policies, no LP optimizer: all fp32, fp16, none, fp16_none
#   ./train.sh lp           the LP optimizer: memory budgets x error limits
#   ./train.sh ablation     re-plan frequency, plan once, error metric, policy subsets,
#                           half_softmax, cost of the error estimate
#   ./train.sh all          baseline, lp and ablation, then the summary
#   ./train.sh summary      only the comparison table (time, memory, error)
#   ./train.sh list         print the commands without running them
#   ./train.sh RUN_NAME...  only these runs (e.g. ./train.sh base_fp32 lp_b55_e0.01)
#
# Every run trains the same model on the same data from the same seed and writes
# $OUT/<run>.json (the training log: plans, per-epoch loss, time and peak memory, test
# perplexity) and $OUT/<run>.txt (the console output). A run whose log exists is
# skipped (FORCE=1 runs it again), so the script can be stopped and resumed; a run
# that fails leaves $OUT/<run>.json.partial and any earlier complete log untouched. The LP
# runs share one profile of the model ($OUT/profile_<size>[_hs].json, made by the
# first of them), so every plan of a sweep comes from the same cost tables; delete it
# after changing the model or the code.
#
# Settings (environment variables, defaults in brackets):
#   DATA [data/ptb]  DEVICE [cuda]  SIZE [activations]  EPOCHS [10]  OPTIMIZER [adam]
#   LR [0.001]  SEED [0]  OUT [stats/experiments]
#   BUDGET [0.55]    reference memory budget, fraction of the all-fp32 peak
#   EPS [0.01]       reference error limit: RMS gradient error / batch noise
#   EPS_LOSS [0.0008] error limit for the loss-metric ablation (loss errors are ~10x smaller)
#   K [5]            re-plan every K epochs
#   VERBOSE [0]      1: also compute the training accuracy every step (slower; the loss
#                    is logged either way)
#   EXTRA []         more apps/train_policies.py arguments for every run, e.g. a smaller
#                    model for a quick check:
#                    EXTRA="--layers 2 --embedding 32 --hidden 64 --batch 4 --seq-len 16" \
#                    DEVICE=cpu EPOCHS=2 ./train.sh all
#   FORCE [0]  DRY_RUN [0]
set -uo pipefail
cd "$(dirname "$0")"

DATA=${DATA:-data/ptb}
DEVICE=${DEVICE:-cuda}
SIZE=${SIZE:-activations}
EPOCHS=${EPOCHS:-10}
OPTIMIZER=${OPTIMIZER:-adam}
LR=${LR:-0.001}
SEED=${SEED:-0}
OUT=${OUT:-stats/experiments}
BUDGET=${BUDGET:-0.55}
EPS=${EPS:-0.01}
EPS_LOSS=${EPS_LOSS:-0.0008}
K=${K:-5}
VERBOSE=${VERBOSE:-0}
EXTRA=${EXTRA:-}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
export PYTHONUNBUFFERED=1

COMMON=(--data "$DATA" --device "$DEVICE" --size "$SIZE" --epochs "$EPOCHS" --optimizer "$OPTIMIZER"
        --lr "$LR" --seed "$SEED")
[[ "$VERBOSE" == 1 ]] && COMMON+=(--verbose)
read -r -a EXTRA_ARGS <<< "$EXTRA"
PROFILE_HS="$OUT/profile_${SIZE}_hs.json"
PROFILE_NOHS="$OUT/profile_${SIZE}.json"
FAILED=()
SELECTED=()

pct() { awk -v b="$1" 'BEGIN { printf "%02d", b * 100 + 0.5 }'; }

# run NAME ARGS...: one training run
run() {
  local name=$1; shift
  if (( ${#SELECTED[@]} )) && [[ ! " ${SELECTED[*]} " == *" $name "* ]]; then return; fi
  local log="$OUT/$name.json"
  if [[ -f "$log" && "$FORCE" != 1 && "$DRY_RUN" != 1 ]]; then
    echo "skip $name ($log exists; FORCE=1 to run it again)"
    return
  fi
  # written as .partial and renamed when the run succeeds, so that a failed or
  # interrupted run never replaces (or looks like) a complete log
  local partial="$log.partial"
  local cmd=(python apps/train_policies.py "${COMMON[@]}" "$@" "${EXTRA_ARGS[@]}" --log "$partial")
  echo ">>> $name: ${cmd[*]}"
  [[ "$DRY_RUN" == 1 ]] && return
  mkdir -p "$OUT"
  if "${cmd[@]}" 2>&1 | tee "$OUT/$name.txt"; then
    mv "$partial" "$log"
  else
    echo "!!! $name failed (see $OUT/$name.txt; partial log $partial)"
    FAILED+=("$name")
  fi
}

# lp NAME ARGS...: a run with the LP optimizer (half_softmax, shared profile, re-plan every K)
lp() {
  local name=$1; shift
  local profile=(--save-profile "$PROFILE_HS")
  [[ -f "$PROFILE_HS" ]] && profile=(--profile "$PROFILE_HS")
  run "$name" --half-softmax --replan-every "$K" "${profile[@]}" "$@"
}

baseline() {
  # no LP optimizer: the same policy for every layer, for the whole training
  run base_fp32      --half-softmax --static fp32       # the reference (time, memory, loss)
  run base_fp16      --half-softmax --static fp16       # mixed precision only (like AMP)
  run base_none      --half-softmax --static none       # gradient checkpointing only
  run base_fp16_none --half-softmax --static fp16_none  # both, everywhere
}

lp_runs() {
  # memory budget sweep at the reference error limit
  for b in 0.7 "$BUDGET" 0.45; do
    lp "lp_b$(pct "$b")_e$EPS" --budget-fraction "$b" --eps "$EPS"
  done
  # error limit sweep at the reference budget
  for e in 0.005 0.02; do
    lp "lp_b$(pct "$BUDGET")_e$e" --budget-fraction "$BUDGET" --eps "$e"
  done
  lp "lp_b$(pct "$BUDGET")_noeps" --budget-fraction "$BUDGET"  # memory only: no error limit
  lp "lp_b35_noeps" --budget-fraction 0.35                     # a budget only fp16_none reaches
  lp "lp_nobudget_e$EPS" --eps "$EPS"                          # error only: the fastest within eps
}

ablation() {
  local ref=(--budget-fraction "$BUDGET" --eps "$EPS")
  # how often to re-plan: every epoch, or only once (a static plan from epoch 0)
  lp abl_k1        "${ref[@]}" --replan-every 1
  lp abl_plan_once "${ref[@]}" --replan-every 1000000
  # the loss error (first-order loss change) instead of the gradient error
  lp abl_metric_loss --budget-fraction "$BUDGET" --eps "$EPS_LOSS" --metric loss
  # smaller policy sets: precision only, checkpointing only, without fp16_none
  lp abl_precision_only     "${ref[@]}" --policies fp32,fp16
  lp abl_checkpoint_only    --budget-fraction "$BUDGET" --policies fp32,none
  lp abl_without_fp16_none  "${ref[@]}" --policies fp32,fp16,none
  # the error estimate's gradient term on all 8 batches instead of 2 (cost vs plan)
  lp abl_grad_batches8      "${ref[@]}" --grad-batches 8
  # attention softmax of float16 layers in float32 (its own profile: different costs)
  local profile=(--save-profile "$PROFILE_NOHS")
  [[ -f "$PROFILE_NOHS" ]] && profile=(--profile "$PROFILE_NOHS")
  run abl_no_half_softmax "${ref[@]}" --replan-every "$K" "${profile[@]}"
  run abl_fp16_no_half_softmax --static fp16
}

summary() {
  [[ "$DRY_RUN" == 1 ]] && return
  python apps/summarize_runs.py "$OUT" --reference base_fp32 --csv "$OUT/summary.csv"
}

finish() {
  if (( ${#FAILED[@]} )); then
    echo "failed runs: ${FAILED[*]}"
    exit 1
  fi
}

case "${1:-all}" in
  baseline) baseline; finish ;;
  lp) lp_runs; finish ;;
  ablation) ablation; finish ;;
  all) baseline; lp_runs; ablation; summary; finish ;;
  summary) summary ;;
  list) DRY_RUN=1; baseline; lp_runs; ablation ;;
  -h|--help|help) sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' ;;
  *) SELECTED=("$@"); baseline; lp_runs; ablation; finish ;;
esac
