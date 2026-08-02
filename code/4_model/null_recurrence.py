#!/usr/bin/env python3
"""Null for cross-tissue recurrence of permutation-validated per-drug signatures.

The per-drug sweep calls a (disease, drug) panel significant at perm_p < 0.05.
Some drugs are called in several independent tissues, and the manuscript argues
that this recurrence is hard to explain by chance. That argument needs a null,
because recurrence is not rare by itself: a drug tested in many tissues has many
chances to be called, and tissues differ in how many hits they yield.

Null design (label-shuffling that preserves both marginals):
  * per-tissue hit counts are held fixed, and
  * each drug can only be called in tissues where it was actually tested,
so the only thing broken is WHICH of a tissue's tested drugs are the hits.
For each permutation, hits are redrawn uniformly without replacement from each
tissue's tested drugs, then drugs recurring in >= k tissues are counted.

Run:  /opt/miniconda3/envs/ML/bin/python 4_model/null_recurrence.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "out" / "signature_results_f2_signatures.csv"
MODEL = "signature"      # the in-fold signature arm; "mean"/"enet" are its baselines,
                         # so filter to one row per (disease, drug) before counting hits.
ALPHA = 0.05
N_PERM = 2000
SEED = 0
KS = (2, 3, 4, 5)


def recurrence_counts(hit_drugs_by_disease: dict, ks=KS) -> dict:
    tally: dict[str, int] = {}
    for drugs in hit_drugs_by_disease.values():
        for dr in drugs:
            tally[dr] = tally.get(dr, 0) + 1
    n = np.fromiter(tally.values(), dtype=int, count=len(tally))
    return {k: int((n >= k).sum()) for k in ks}


def _report(label: str, observed: dict, null: dict) -> None:
    print(f"\n{label}")
    print(f"{'k':>3}  {'observed':>8}  {'null mean':>9}  {'null p95':>8}  {'z':>7}  {'perm p':>8}")
    for k in KS:
        v = null[k]
        z = (observed[k] - v.mean()) / v.std(ddof=1) if v.std(ddof=1) > 0 else np.nan
        p = (1 + (v >= observed[k]).sum()) / (N_PERM + 1)
        print(
            f"{k:>3}  {observed[k]:>8}  {v.mean():>9.1f}  {np.percentile(v, 95):>8.1f}  "
            f"{z:>7.1f}  {p:>8.4f}"
        )


def main() -> None:
    d = pd.read_csv(SWEEP)
    d = d[d.model == MODEL]
    hits = d[d.perm_p < ALPHA]
    print(f"panels {len(d)}, hits at p < {ALPHA}: {len(hits)}")
    observed = recurrence_counts({ds: g.drug_name.tolist() for ds, g in hits.groupby("disease")})

    # --- null 1: hits redrawn uniformly from each tissue's tested drugs ---------
    tested = {ds: g.drug_name.to_numpy() for ds, g in d.groupby("disease")}
    n_hits = hits.groupby("disease").size().to_dict()
    rng = np.random.default_rng(SEED)
    null = {k: np.empty(N_PERM, dtype=int) for k in KS}
    for i in range(N_PERM):
        drawn = {ds: rng.choice(tested[ds], size=n_hits.get(ds, 0), replace=False) for ds in tested}
        c = recurrence_counts(drawn)
        for k in KS:
            null[k][i] = c[k]
    _report("null 1: uniform within tissue", observed, null)

    # --- null 2: additionally matched on panel size (power) --------------------
    # A drug screened in large panels has more power to be called in every tissue,
    # so recurrence could reflect sample size rather than shared biology. Drawing
    # hits within n_lines quartiles holds that profile fixed.
    d = d.copy()
    d["bin"] = d.groupby("disease").n_lines.transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False)
    )
    pools = {key: g.drug_name.to_numpy() for key, g in d.groupby(["disease", "bin"])}
    per_bin = d[d.perm_p < ALPHA].groupby(["disease", "bin"]).size()
    rng = np.random.default_rng(SEED)
    null2 = {k: np.empty(N_PERM, dtype=int) for k in KS}
    for i in range(N_PERM):
        drawn = {
            key: rng.choice(pools[key], size=min(cnt, len(pools[key])), replace=False)
            for key, cnt in per_bin.items()
        }
        c = recurrence_counts(drawn)
        for k in KS:
            null2[k][i] = c[k]
    _report("null 2: matched on panel size within tissue", observed, null2)


if __name__ == "__main__":
    main()
