# Dynamic Mixed-Precision and Recomputation Policies for needle

A 10-714 (Deep Learning Systems) final project. We extend needle, the course's deep learning
library, so that every transformer layer chooses its own policy: float32 or float16, and
activations stored or recomputed (gradient checkpointing). A profiler measures what each
choice costs in time and memory, an estimate measures its precision error relative to the
gradient noise, and a mixed-integer linear program picks the fastest configuration within
a memory budget and an error limit. A controller repeats the choice during training.

The report is [`final_report.ipynb`](final_report.ipynb). It explains the design, shows the
results and runs small demonstrations.

## Contents

| Path | What it holds |
|---|---|
| `src/` | C++ and CUDA kernels (float32, float16, bfloat16 storage; tensor-core matmul) |
| `python/needle/` | the needle library: autograd with casts and `checkpoint()`, `amp.py` (autocast, `GradScaler`), `nn/` (the transformer with per-layer policies) |
| `python/needle/policy/` | the optimizer: `profiler.py`, `error.py`, `solver.py` (MILP), `controller.py` |
| `apps/` | command-line tools: `train_policies.py` (training), `profile_segments.py`, `estimate_errors.py`, `plan_policies.py`, `summarize_runs.py` (results table) |
| `train.sh`, `plot.py` | the experiment runs of the report, and its figures |
| `stats/experiments/` | the training logs of every run in the report, and the GPU profile they share |
| `tests/` | the test suite |
| `human-docs/` | the user guide of the optimizer (`policy-optimizer/user-guide.md`) and design notes |

Run every command below from the root of the repository.

## 1. Prepare the environment

You need:

- Linux, Python 3.10 or newer (we used 3.13), a C++ compiler and CMake 3.27 or newer;
- for the experiments: an NVIDIA GPU and the CUDA toolkit (`nvcc`). The float16 tensor-core
  matmul needs compute capability 7.0 or higher (8.0 for bfloat16). Without CUDA, only the
  CPU backend is built; that is enough for the tests, the quick check and the figures.

Create an environment, install the packages and build the backends:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
make
```

`make` builds `python/needle/backend_ndarray/ndarray_backend_cpu*.so` and, when CMake finds
CUDA, `ndarray_backend_cuda*.so`. If your CMake is older than 3.27, install a newer one with
`pip install "cmake>=3.27"` and run `make` again. After installing CUDA later, rebuild with
`make clean && make`.

Check which backends are available:

```bash
python -c "import sys; sys.path.insert(0, 'python'); import needle as ndl; print('cpu', ndl.cpu().enabled(), 'cuda', ndl.cuda().enabled())"
```

Run the tests (about 3 minutes on a CPU; the GPU tests are skipped when there is no GPU):

```bash
python -m pytest tests
```

### Data

The experiments use Penn Treebank as `data/ptb/train.txt` and `data/ptb/test.txt`:

```bash
mkdir -p data/ptb
```

```bash
wget -nc https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt -O data/ptb/train.txt
```

```bash
wget -nc https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt -O data/ptb/test.txt
```

## 2. Reproduce the results

### 2.1 The table and the figures, from the saved logs (no GPU needed, seconds)

The logs of all 16 runs of the report are in `stats/experiments/`. The results table
(time, memory, loss and perplexity of every run, relative to all-float32):

```bash
python apps/summarize_runs.py stats/experiments --reference base_fp32
```

The figures of the report, written to `stats/experiments/plots/`:

```bash
python plot.py
```

### 2.2 A quick check of the whole pipeline (CPU, no data, under a minute)

A small model on random tokens: the controller profiles the model, estimates the precision
errors, solves the MILP and re-plans after every epoch.

```bash
python apps/train_policies.py --data random --device cpu --layers 2 --embedding 32 --hidden 64 --batch 4 --seq-len 16 --random-steps 6 --epochs 2 --replan-every 1 --budget-fraction 0.7 --eps 0.05
```

### 2.3 The full experiments (GPU and Penn Treebank, about 6 hours)

`train.sh` runs the 16 experiments of the report one after the other: four fixed policies
without the planner (`base_*`), the budget and error-limit sweeps (`lp_*`) and the ablations
(`abl_*`). All of them train the same 4-layer transformer for 10 epochs with the same seed.
Each run takes about 20 minutes on our GPU.

The script skips every run whose log already exists, and the logs of the report are in
`stats/experiments`. To run the experiments again, write into a new folder:

```bash
OUT=stats/rerun ./train.sh all
```

```bash
OUT=stats/rerun ./train.sh plot
```

`all` ends with the results table (also saved as `$OUT/summary.csv`); `plot` writes the
figures to `$OUT/plots`. More ways to run it:

| Command | What it does |
|---|---|
| `OUT=stats/rerun ./train.sh list` | print the commands without running them |
| `OUT=stats/rerun ./train.sh baseline` | only the fixed policies (also `lp`, `ablation`) |
| `OUT=stats/rerun ./train.sh base_fp32 lp_b60_e0.01` | only the named runs |
| `OUT=stats/rerun ./train.sh summary` | only the results table |

- Run the script as `./train.sh`, not with `source`.
- A stopped run leaves `<run>.json.partial`; start the script again to continue with the
  missing runs. `FORCE=1` runs existing ones again.
- Settings are environment variables: `DEVICE` (cuda), `EPOCHS` (10), `BUDGET` (0.6, the
  reference memory budget as a fraction of the all-float32 peak), `EPS` (0.01, the reference
  error limit), `K` (5, re-plan every K epochs), `DATA` (data/ptb), `VERBOSE` (0; 1 also
  computes the training accuracy) and `EXTRA` (more arguments for every run). The header of
  `train.sh` lists them all.
- The first planned run profiles the model (about 2 minutes) and saves the profile as
  `$OUT/profile_activations_hs.json`; the other planned runs reuse it. Times depend on the
  GPU, so absolute numbers differ on other hardware. On the same kind of GPU, copying
  `stats/experiments/profile_activations_hs.json` into `$OUT` first reuses our profile.

Each run writes `$OUT/<run>.json` (plans, per-epoch loss, time and peak memory, test
perplexity) and `$OUT/<run>.txt` (the console output).

### 2.4 The report notebook

Open `final_report.ipynb` in Jupyter or Google Colab. Its outputs are saved, so it reads
without running anything. Run from the repository root, its cells use this checkout. In
Colab, the setup cells download the code archive from the link in the notebook, install the
packages and build needle.

## Using the optimizer on your own runs

The planner can also be run step by step: profile, estimate the errors, plan, train. The
tools and all their options are described in
[`human-docs/policy-optimizer/user-guide.md`](human-docs/policy-optimizer/user-guide.md).
