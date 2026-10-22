"""Amendment 6 criterion (frozen before any amendment 6 evaluation): N = 100, fine-tuned official GENCO versus scratch,
training seeds 0-4, sorted test positions 1000-2999, amendment 5 gate.

    python scripts/amendment6_analysis.py          # reads results/genco_opf_amendment6_seed{0..4}.json

Success requires all four:
  1. mean over seeds of the fine-tuned hosting share >= 0.95 (of the raw IPOPT reference);
  2. mean over seeds of (fine-tuned - scratch) >= 2 percentage points;
  3. paired two-level bootstrap (seeds with replacement, then scenarios with replacement, the same scenarios for both
     arms of a seed) of that mean difference: 2.5th percentile > 0;
  4. fine-tuned > scratch in at least 4 of 5 seeds.
"""
import json
from pathlib import Path

import numpy as np

SEEDS, N_BOOT, BOOT_SEED = range(5), 10000, 6
A, B = "a5_finetune_N100", "a5_scratch_N100"


def main():
    res = [json.loads(Path(f"results/genco_opf_amendment6_seed{s}.json").read_text()) for s in SEEDS]
    ids = {r["test_ids_sha256"] for r in res}
    assert len(ids) == 1 and all(r["test_positions"] == [1000, 3000] for r in res), "seeds were not evaluated on the same scenarios"
    ref = np.array(res[0]["per_scenario"]["reference_flex_mw"])
    assert all(np.array_equal(ref, r["per_scenario"]["reference_flex_mw"]) for r in res)
    ha = np.array([r["per_scenario"][A]["hosting_mw"] for r in res])      # (seeds, scenarios)
    hb = np.array([r["per_scenario"][B]["hosting_mw"] for r in res])
    share_a, share_b = ha.sum(1) / ref.sum(), hb.sum(1) / ref.sum()
    diff = 100 * (share_a - share_b)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(N_BOOT)
    for k in range(N_BOOT):
        s = rng.integers(0, len(res), len(res))
        boot[k] = np.mean([100 * (ha[i, j].sum() - hb[i, j].sum()) / ref[j].sum()
                           for i, j in ((i, rng.integers(0, len(ref), len(ref))) for i in s)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    c = {"per_seed": [{"seed": s, "finetuned_share": float(a), "scratch_share": float(b), "diff_points": float(d)}
                      for s, a, b, d in zip(SEEDS, share_a, share_b, diff)],
         "mean_finetuned_share": float(share_a.mean()), "mean_scratch_share": float(share_b.mean()),
         "mean_diff_points": float(diff.mean()), "boot_lo": float(lo), "boot_hi": float(hi),
         "seeds_finetuned_better": int((diff > 0).sum()), "test_ids_sha256": ids.pop()}
    c["c1_share_ge_95"] = c["mean_finetuned_share"] >= 0.95
    c["c2_diff_ge_2"] = c["mean_diff_points"] >= 2.0
    c["c3_lo_gt_0"] = c["boot_lo"] > 0
    c["c4_ge_4_of_5"] = c["seeds_finetuned_better"] >= 4
    c["pass"] = bool(c["c1_share_ge_95"] and c["c2_diff_ge_2"] and c["c3_lo_gt_0"] and c["c4_ge_4_of_5"])
    Path("results/genco_opf_amendment6.json").write_text(json.dumps(c, indent=1) + "\n")
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
