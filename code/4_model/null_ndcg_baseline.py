#!/usr/bin/env python3
"""Random-ranking null for the drug-ranking metric used in Figure S3.

`evaluate.ranking_metrics` scores NDCG@k with graded relevance (max - sens_z) and
k=3, averaged over cell lines. NDCG has a non-zero expectation under a random
ranking, so "the pan-cancer value collapses to chance" needs an explicit null.

This permutes each cell line's own responses (breaking the correspondence between
a drug and its measured sensitivity while preserving that line's sensitivity
distribution exactly) and reports the resulting NDCG@3 distribution.

Run:  /opt/miniconda3/envs/ML/bin/python 4_model/null_ndcg_baseline.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import ndcg_score

HERE = Path(__file__).resolve().parent
RESP = HERE.parent / "1_harmonize" / "out" / "drug_response_long.parquet"
K = 3
MIN_DRUGS = 10          # matches the >=10-measured-drugs floor used when scoring lines
N_PERM = 100            # permutations per cell line
SEED = 0


def main() -> None:
    d = pd.read_parquet(RESP, columns=["depmap_id", "drug_uid", "sens_z"]).dropna()
    lines = [g["sens_z"].to_numpy() for _, g in d.groupby("depmap_id") if len(g) >= MIN_DRUGS]
    rng = np.random.default_rng(SEED)

    per_line = []
    for yt in lines:
        rel_true = (yt.max() - yt)[None, :]
        vals = []
        for _ in range(N_PERM):
            yp = rng.permutation(yt)
            vals.append(ndcg_score(rel_true, (yp.max() - yp)[None, :], k=min(K, len(yt))))
        per_line.append(np.mean(vals))

    v = np.asarray(per_line)
    print(f"cell lines scored (>= {MIN_DRUGS} drugs): {len(v)}")
    print(f"permutations per line: {N_PERM}")
    print(
        f"random-ranking NDCG@{K}: mean {v.mean():.3f}  median {np.median(v):.3f}  "
        f"p5 {np.percentile(v, 5):.3f}  p95 {np.percentile(v, 95):.3f}"
    )


if __name__ == "__main__":
    main()
