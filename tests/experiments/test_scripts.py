"""The experiment tools: apps/summarize_runs.py and train_policies.py --save-profile."""
import json
import sys

sys.path.append("./python")
sys.path.append(".")
import pytest

MIB = 2 ** 20


def write_log(path, static=None, budget_fraction=None, eps=None, ms=(100, 100), peak=1000, losses=(5.0, 4.0),
              ppl=300.0, replans=()):
    epochs = [{"epoch": i, "policies": ["fp32", "fp32"] if static else ["none", "fp16"], "loss": l, "acc": None,
               "seconds": m * 10 / 1e3, "steps": 10, "measured_peak": peak * MIB, "predicted_peak": None,
               "over_budget": False, "skipped_steps": 0} for i, (m, l) in enumerate(zip(ms, losses))]
    log = {"config": {"static_policies": static, "budget": None, "budget_fraction": budget_fraction, "eps": eps,
                      "metric": "grad", "replan_every": 5, "policies": None, "grad_batches": 2, "half_softmax": True},
           "replans": list(replans), "epochs": epochs, "results": {"test_perplexity": ppl}}
    path.write_text(json.dumps(log))


def test_summarize_runs(tmp_path, capsys):
    from apps.summarize_runs import main

    write_log(tmp_path / "base_fp32.json", static=["fp32", "fp32"], ms=(100, 102, 98), peak=1000,
              losses=(5.0, 4.0, 3.0), ppl=300.0)
    replan = {"epoch": 0, "policies": ["none", "fp16"], "feasible": False, "predicted_error": 0.008,
              "seconds": {"profile": 10.0, "errors": 2.0, "solve": 0.5}}
    write_log(tmp_path / "lp_run.json", budget_fraction=0.5, eps=0.01, ms=(80, 80, 80), peak=400,
              losses=(5.1, 4.0, 3.03), ppl=303.0, replans=[replan])
    (tmp_path / "profile_activations_hs.json").write_text(json.dumps({"policies": ["fp32"]}))  # not a run
    (tmp_path / "broken.json.partial").write_text("{")
    rows = main([str(tmp_path), "--reference", "base_fp32", "--csv", str(tmp_path / "summary.csv")])
    assert set(rows) == {"base_fp32", "lp_run"}
    r = rows["lp_run"]
    assert r["mode"] == "LP b=0.5 eps=0.01 k=5" and rows["base_fp32"]["mode"] == "fixed fp32"
    assert r["ms_step"] == 80 and r["speedup"] == pytest.approx(100 / 80)  # medians
    assert r["mem_saved"] == pytest.approx(0.6) and r["peak_mib"] == 400
    assert r["d_loss"] == pytest.approx(0.01) and r["max_d_loss"] == pytest.approx(0.02)
    assert r["d_ppl"] == pytest.approx(0.01) and r["pred_error"] == 0.008
    assert r["planning_s"] == pytest.approx(12.5) and r["notes"] == "infeasible 1/1"
    out = capsys.readouterr().out
    assert out.index("| base_fp32") < out.index("| lp_run")  # the reference first
    assert (tmp_path / "summary.csv").read_text().count("\n") == 3


def test_save_and_reuse_a_profile(tmp_path, monkeypatch):
    from apps.train_policies import main
    from needle.policy import SegmentProfile
    from needle.policy import controller as controller_module

    common = ["--data", "random", "--device", "cpu", "--layers", "2", "--embedding", "32", "--hidden", "64",
              "--batch", "4", "--seq-len", "12", "--random-steps", "4", "--epochs", "1", "--budget-fraction", "0.9"]
    path = tmp_path / "profile.json"
    first = main(common + ["--save-profile", str(path)])
    saved = SegmentProfile.load(str(path))
    assert saved == first.profile and saved.key == "12x4"

    def no_profiling(*a, **k):
        raise AssertionError("profiled although --profile was given")

    monkeypatch.setattr(controller_module, "profile_segments", no_profiling)
    second = main(common + ["--profile", str(path)])
    assert second.profile == saved and not second.history[0]["profiled"]
    assert second.plan.policies == first.plan.policies  # the same cost table, the same plan


def test_plot_every_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from plot import FIGURES, main

    def replan(epoch, policies, feasible=True):
        return {"epoch": epoch, "policies": policies, "feasible": feasible, "predicted_error": 0.008,
                "segment_errors": [0.006, 0.009], "noise": 10.0 / (1 + epoch), "mean_loss": 5.0,
                "seconds": {"profile": 1.0, "errors": 2.0, "solve": 0.1}}

    def lp_log(budget, eps, policies, feasible=True, ppl=310.0, ms=90.0):
        epochs = [{"epoch": i, "policies": p, "loss": 5.0 - i / 10, "seconds": ms * 10 / 1e3, "steps": 10,
                   "measured_peak": 600 * MIB, "predicted_peak": 598 * MIB, "predicted_step_time": 0.088,
                   "over_budget": False, "skipped_steps": 0} for i, p in enumerate(policies)]
        return {"config": {"static_policies": None, "budget_fraction": budget, "eps": eps, "metric": "grad",
                           "replan_every": 2, "policies": None, "grad_batches": 2, "half_softmax": True},
                "replans": [replan(i, policies[i], feasible) for i in range(0, len(policies), 2)],
                "epochs": epochs, "results": {"test_perplexity": ppl}}

    write_log(tmp_path / "base_fp32.json", static=["fp32", "fp32"], ms=(100, 100, 100, 100),
              losses=(5.0, 4.9, 4.8, 4.7))
    write_log(tmp_path / "base_fp16.json", static=["fp16", "fp16"], ms=(80, 80, 80, 80), peak=800,
              losses=(5.0, 4.9, 4.8, 4.71), ppl=305.0)
    plan = [["fp32", "none"], ["fp32", "none"], ["fp16", "fp16_none"], ["fp16", "fp16_none"]]
    for name, budget, eps, feasible in [("lp_b40_e0.01", 0.4, 0.01, False), ("lp_b60_e0.01", 0.6, 0.01, True),
                                        ("lp_b60_e0.02", 0.6, 0.02, True), ("lp_b60_noeps", 0.6, None, True),
                                        ("abl_k1", 0.6, 0.01, True)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(lp_log(budget, eps, plan, feasible)))
    out = tmp_path / "plots"
    written = main([str(tmp_path), "--out", str(out)])
    assert sorted(written) == sorted(str(out / f"{f}.png") for f in FIGURES)
    assert all((out / f"{f}.png").stat().st_size > 10_000 for f in FIGURES)
    # without the reference run the relative figures are skipped, the rest still drawn
    written = main([str(tmp_path / "lp_b60_e0.01.json"), "--out", str(out / "one"), "--theme", "dark"])
    assert str(out / "one" / "plans.png") in written and str(out / "one" / "loss.png") not in written
