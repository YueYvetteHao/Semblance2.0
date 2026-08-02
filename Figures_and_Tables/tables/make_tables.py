#!/usr/bin/env python3
"""Generate the manuscript's markdown tables from the figure data folders.

Writes tables/Table1.md, tables/Table2.md, tables/TableS1.md. build_v3.py splices
them in at the {{TABLE_N}} placeholders in the section markdowns, so the numbers in
the tables and the numbers in the figures always come from the same source files.

Table3.md is NOT generated here and is not overwritten by this script. It is a curated
selection — one representative panel per drug, chosen for mechanistic coherence — rather
than a threshold applied to a result table, so it is maintained by hand and lives inline
in the Results section of the manuscript. The full 971-panel validated set it is drawn
from is in Figure3/data/validated_hits.csv and the released signature atlas.

Run:  /opt/miniconda3/envs/ML/bin/python make_tables.py
"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent
OUT = D / "tables"
OUT.mkdir(exist_ok=True)

DISEASE_NAME = {
    "ALL": "Acute lymphoblastic leukaemia", "BLCA": "Bladder", "BRCA": "Breast",
    "COAD/READ": "Colorectal", "DLBC": "Diffuse large B-cell lymphoma", "ESCA": "Oesophageal",
    "GBM": "Glioblastoma", "HNSC": "Head and neck", "KIRC": "Renal", "LAML": "Acute myeloid leukaemia",
    "LIHC": "Liver", "LUAD": "Lung adenocarcinoma", "OV": "Ovarian", "PAAD": "Pancreatic",
    "SARC": "Sarcoma", "SCLC": "Small-cell lung", "SITE_LUNG": "Lung (pooled)", "SKCM": "Melanoma",
    "STAD": "Gastric", "UCEC": "Endometrial",
}


def table1() -> str:
    h = pd.read_csv(D / "FigureS1" / "data" / "harmonization_counts.csv")
    rows = [
        "### Table 1. The harmonized GDSC∪CTRP drug-response resource",
        "",
        "Cell lines and compounds contributed by each screen, their union, and the "
        "structure-confirmed intersection used for cross-cohort quality control. Drug identity "
        "is the InChIKey connectivity block, not the compound name.",
        "",
        "| Set | Cell lines | Drugs |",
        "|-----|--:|--:|",
    ]
    for _, r in h.iterrows():
        rows.append(f"| {r['set']} | {int(r.cell_lines):,} | {int(r.drugs):,} |")
    return "\n".join(rows) + "\n"


def table2() -> str:
    a = pd.read_csv(D / "Figure2" / "data" / "ablation_results_v1.csv")
    a = a[a.cohort == "both"]
    dm = a[a.model == "drugmean"].set_index("disease")
    me = a[a.model == "enet_resid"].set_index("disease")
    rows = [
        "### Table 2. Per-disease decomposition: broad-spectrum versus mechanism-specific",
        "",
        "Held-out Spearman correlation (pooled GDSC∪CTRP) for the drug-mean baseline, which ignores "
        "molecular features, and for the expression mechanism arm, which predicts drug-mean-residualized "
        "sensitivity. Brackets give the 95% repeated-cross-validation interval. The mechanism arm clears "
        "zero in one disease only (sarcoma).",
        "",
        "| Disease | Cell lines | Broad-spectrum ρ [95% CI] | Mechanism ρ [95% CI] |",
        "|---------|--:|--:|--:|",
    ]
    for ds in dm.sort_values("spearman", ascending=False).index:
        if ds not in me.index:
            continue
        d, m = dm.loc[ds], me.loc[ds]
        rows.append(
            f"| {DISEASE_NAME.get(ds, ds)} ({ds}) | {int(d.n_lines)} | "
            f"{d.spearman:+.3f} [{d.spearman_lo:+.3f}, {d.spearman_hi:+.3f}] | "
            f"{m.spearman:+.3f} [{m.spearman_lo:+.3f}, {m.spearman_hi:+.3f}] |"
        )
    return "\n".join(rows) + "\n"


def table_s1() -> str:
    e = pd.read_csv(D / "Figure3" / "data" / "canonical_enrichment.csv")
    k, n = int(e.hit.sum()), len(e)
    rows = [
        "### Table S1. Prospective recovery of established drug–target pharmacology",
        "",
        "Every screened combination of seven drug classes with the tissues in which that class's target "
        "is an established dependency. The reference set was fixed from target biology before any result "
        "was consulted, so it is not the set of pairs quoted in Section 3.5, which were selected after "
        f"validating. {k} of {n} panels validate ({100 * k / n:.0f}%) against a background rate of 12.4% "
        "across all 7,836 panels (4.3-fold, one-sided binomial *p* = 1.5e-12). A panel counts as validated "
        "when its permutation *p* < 0.05; every failing panel is listed so the misses are visible.",
        "",
        # Pandoc sets LaTeX column widths from the relative width of these dash runs,
        # so the separator row is the layout control: narrow class, narrow count, wide misses.
        "| Class | Validated | Misses (drug, tissue, permutation *p*) |",
        "|:---------------------|:-------:|:" + "-" * 70 + "|",
    ]
    for cls, g in e.groupby("drug_class"):
        miss = g[~g.hit].sort_values("perm_p")
        txt = "; ".join(f"{r.drug} ({r.tissue}, {r.perm_p:.2f})" for _, r in miss.iterrows()) or "none"
        rows.append(f"| {cls} | {int(g.hit.sum())} / {len(g)} | {txt} |")
    rows += ["", f"**Total: {k} / {n} ({100 * k / n:.0f}%) versus 12.4% background.**", ""]
    return "\n".join(rows) + "\n"


def main() -> None:
    for name, fn in [("Table1", table1), ("Table2", table2), ("TableS1", table_s1)]:
        (OUT / f"{name}.md").write_text(fn())
        print(f"wrote tables/{name}.md")


if __name__ == "__main__":
    main()
