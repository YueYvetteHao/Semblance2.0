"""Emit the JSON shard list for an Argo fan-out — one entry per independently-runnable unit.

Reuses the existing task/pair enumerators so the shard list matches exactly what the monolithic
drivers would iterate over — no re-derivation, no drift:

  pathway_ablation   {disease, cohort}      run_ablation.build_tasks    run_ablation.py --shard
  f2_perm            {disease, drug_uid}    run_signature.build_pairs   run_signature.py --shard
  atlas              {disease}              data.disease_sizes          run_gsea_signatures.R (R; deferred)

Prints a JSON array to stdout (a status line goes to stderr), so it composes cleanly:

    python emit_shards.py --workload pathway_ablation > shards.json
    argo submit fanout.yaml -p shards="$(cat shards.json)" ...

Each array element is one shard object; Argo `withParam` fans one pod per element. Over-emitting
is harmless: a driver's --shard run still writes an ok=False checkpoint for a too-small unit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D               # noqa: E402
import run_ablation as RA      # noqa: E402
import run_signature as RS     # noqa: E402


def _diseases(arg: str) -> list[str] | None:
    """Parse --diseases: 'auto'/'' -> None (enumerate from the store), else a comma list."""
    if not arg or arg == "auto":
        return None
    return [d.strip() for d in arg.split(",") if d.strip()]


def pathway_ablation_shards(args) -> list[dict]:
    """One {disease, cohort} per valid ablation panel (default cohort=both, where pathway arms fire)."""
    diseases = _diseases(args.diseases)
    cfg = {
        "min_lines": args.min_lines,
        "diseases": diseases if diseases else ["auto"],
        "pancancer": args.pancancer,
        "cohorts": [c.strip() for c in args.cohorts.split(",") if c.strip()],
    }
    tasks, _ = RA.build_tasks(cfg)
    return [{"disease": d, "cohort": c} for d, c in tasks]


def _even_chunks(items: list, batch_size: int) -> list[list]:
    """Split `items` into ceil(n/batch_size) near-EQUAL chunks (sizes differ by <=1, each <=batch_size).

    Balanced, not fixed-batch-with-a-ragged-remainder, so every pod is ~the same size: e.g. 629 drugs
    at B=25 -> 26 chunks of 24-25, NOT 25x25 + one lonely pod of 4.
    """
    n = len(items)
    k = max(1, math.ceil(n / batch_size))
    q, r = divmod(n, k)
    out, i = [], 0
    for j in range(k):
        size = q + (1 if j < r else 0)
        out.append(items[i:i + size])
        i += size
    return out


def f2_perm_shards(args) -> list[dict]:
    """f2 shards. Default (batch-size 1): one {disease, drug_uid} per (disease, drug) pair passing the
    size/variance filters. With --batch-size B>1: one {disease, drug_uids:[...]} per near-even batch of
    <=B drugs within a disease — so a pod syncs the ~100 MB expression store ONCE (not once per drug)
    and pods stay evenly sized. Both forms are consumed by `run_signature.py --shard`."""
    diseases = _diseases(args.diseases)
    cfg = {
        "diseases": diseases if diseases else ["auto"],
        "drugs": [d.strip() for d in args.drugs.split(",") if d.strip()],
        "min_lines": args.f2_min_lines,
        "min_sd": args.min_sd,
        "per_disease": args.per_disease,
        "max_pairs": args.max_pairs,
    }
    pairs, _ = RS.build_pairs(cfg)
    if args.batch_size <= 1:
        return [{"disease": r["disease"], "drug_uid": r["drug_uid"]} for _, r in pairs.iterrows()]
    shards = []
    for disease, grp in pairs.groupby("disease", sort=False):
        for chunk in _even_chunks(grp["drug_uid"].tolist(), args.batch_size):
            shards.append({"disease": disease, "drug_uids": chunk})
    return shards


def atlas_shards(args) -> list[dict]:
    """One {disease} per disease (the R fgsea step loops that disease's drugs internally)."""
    diseases = _diseases(args.diseases)
    if diseases is None:
        diseases = list(D.disease_sizes(min_lines=args.min_lines).index)
    return [{"disease": d} for d in diseases]


WORKLOADS = {
    "pathway_ablation": pathway_ablation_shards,
    "f2_perm": f2_perm_shards,
    "atlas": atlas_shards,
}


def main():
    ap = argparse.ArgumentParser(description="Emit the JSON shard list for an Argo fan-out.")
    ap.add_argument("--workload", required=True, choices=list(WORKLOADS))
    ap.add_argument("--diseases", default="auto", help="'auto' (enumerate) or comma TCGA codes")
    # pathway_ablation / atlas
    ap.add_argument("--cohorts", default="both",
                    help="pathway_ablation: comma cohorts (default 'both' — where the pathway arms fire)")
    ap.add_argument("--pancancer", action="store_true",
                    help="pathway_ablation: also emit the pooled pan-cancer panel")
    ap.add_argument("--min-lines", type=int, default=25,
                    help="pathway_ablation/atlas: min cells per disease to enumerate it")
    # f2_perm
    ap.add_argument("--drugs", default="", help="f2_perm: comma drug-name substrings to restrict to")
    ap.add_argument("--f2-min-lines", type=int, default=20,
                    help="f2_perm: min lines per (disease, drug) pair")
    ap.add_argument("--min-sd", type=float, default=0.3, help="f2_perm: min sd of sens_z per pair")
    ap.add_argument("--per-disease", type=int, default=0, help="f2_perm: top-N drugs per disease (0 = all)")
    ap.add_argument("--max-pairs", type=int, default=100_000, help="f2_perm: global cap (default: uncapped)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="f2_perm: drugs per pod. 1 = one pod per drug; B>1 chunks each disease into "
                         "ceil(n/B) near-even <=B-drug pods (S3 synced once per pod, not per drug).")
    args = ap.parse_args()

    shards = WORKLOADS[args.workload](args)
    json.dump(shards, sys.stdout)
    sys.stdout.write("\n")
    print(f"# {len(shards)} shards for workload={args.workload}", file=sys.stderr)


if __name__ == "__main__":
    main()
