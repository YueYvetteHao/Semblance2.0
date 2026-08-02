#!/usr/bin/env python3
"""Is the permutation-validated set enriched for established drug-target pharmacology?

The two-stage protocol promotes a panel only on independent corroboration, and the
first corroboration is "the retained set recovers known pharmacology". That is an
enrichment claim and needs a statistic.

The obvious test is circular: the canonical pairs quoted in the Results were quoted
BECAUSE they validated. So the reference set here is fixed by target biology alone,
before looking at any result, as the cross product of a drug class and the tissues
in which that class's target is an established dependency. Drugs of a class that
were screened but never mentioned in the manuscript are included, and pairs that
fail are counted as failures.

Run:  /opt/miniconda3/envs/ML/bin/python canonical_enrichment.py
Writes out/canonical_enrichment.csv
"""
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "out" / "signature_results_f2_signatures.csv"
MODEL = "signature"      # the in-fold signature arm; "mean"/"enet" are its baselines,
                         # so filter to one row per (disease, drug) before counting hits.
ALPHA = 0.05

# class -> (drugs as screened, tissues where the target is an established dependency)
PANEL = {
    "BRAF": (
        ["PLX-4720", "PLX-4032", "Dabrafenib", "dabrafenib"],
        ["SKCM", "COAD/READ"],
    ),
    "MEK": (
        ["Trametinib", "trametinib", "Selumetinib", "selumetinib", "Refametinib"],
        ["SKCM", "COAD/READ", "LUAD", "PAAD"],
    ),
    "EGFR/HER2": (
        ["Erlotinib", "erlotinib", "Gefitinib", "gefitinib", "Afatinib", "afatinib",
         "Lapatinib", "lapatinib"],
        ["LUAD", "BRCA", "HNSC"],
    ),
    "BCR-ABL/SRC": (
        ["Dasatinib", "dasatinib", "Imatinib", "imatinib", "Nilotinib", "nilotinib"],
        ["ALL", "LAML"],
    ),
    "BCL2": (
        ["ABT-199", "Venetoclax", "Navitoclax", "navitoclax"],
        ["ALL", "DLBC", "LAML"],
    ),
    "IGF1R": (
        ["BMS-754807", "Linsitinib", "linsitinib", "NVP-ADW742"],
        ["SARC", "BRCA", "OV"],
    ),
    "MDM2/p53": (
        ["nutlin-3", "Nutlin-3a (-)", "RITA"],
        ["SARC", "ALL", "COAD/READ"],
    ),
}


def main() -> None:
    d = pd.read_csv(SWEEP)
    d = d[d.model == MODEL].copy()
    d["hit"] = d.perm_p < ALPHA
    base = d.hit.mean()

    rows = []
    for cls, (drugs, tissues) in PANEL.items():
        for dr in drugs:
            for ds in tissues:
                s = d[(d.drug_name == dr) & (d.disease == ds)]
                if s.empty:
                    continue                      # not screened in that tissue
                rows.append({
                    "drug_class": cls, "drug": dr, "tissue": ds,
                    "n_lines": int(s.n_lines.iloc[0]),
                    "spearman": float(s.spearman.iloc[0]),
                    "perm_p": float(s.perm_p.iloc[0]),
                    "hit": bool(s.hit.iloc[0]),
                })

    ref = pd.DataFrame(rows)
    k, n = int(ref.hit.sum()), len(ref)
    p = stats.binomtest(k, n, base, alternative="greater").pvalue

    print(f"background hit rate: {base:.4f}  ({d.hit.sum()}/{len(d)} panels)")
    print(f"mechanism-defined reference panels screened: {n}")
    print(f"validated: {k} ({100 * k / n:.0f}%),  {k / n / base:.1f}-fold enrichment,  binomial p = {p:.3g}")
    print("\nby class:")
    print(ref.groupby("drug_class").hit.agg(["sum", "size"]).to_string())
    print("\nmisses:")
    print(ref[~ref.hit][["drug_class", "drug", "tissue", "n_lines", "spearman", "perm_p"]].to_string(index=False))

    out = HERE / "out" / "canonical_enrichment.csv"
    ref.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
