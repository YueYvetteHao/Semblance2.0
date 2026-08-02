"""Drug crosswalk: give every GDSC and CTRP compound one shared identity.

Strategy (structure-first, name/synonym as backup):
  * CTRP ships SMILES -> RDKit InChIKey (local, reliable).
  * GDSC ships no SMILES/CID here -> resolve InChIKey from PubChem by DRUG_NAME,
    falling back through the GDSC `SYNONYMS` list until one resolves. Cached.
  * Match on the InChIKey *connectivity block* (first 14 chars) so stereochemistry
    and salt form don't split a compound. Fall back to normalized-name / synonym.

`drug_uid` is the shared key: the InChIKey block when structure is known, else a
`name:<normname>` slug. Rows of the same compound in both cohorts share it.

Run directly to build/cache `ref/drug_crosswalk.parquet` (+ `ref/pubchem_cache.json`).
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
import pandas as pd
import requests
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent / "raw_data"
REF = HERE / "ref"

GDSC_XLSX = RAW / "GDSC2_fitted_dose_response_27Oct23.xlsx"
GDSC_ANNO = REF / "gdsc_screened_compounds.csv"
CTRP_CPD = RAW / "CTRPv2.2_2015_pub_CancerDisc_5_1210" / "v22.meta.per_compound.txt"
PUBCHEM_CACHE = REF / "pubchem_cache.json"

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def norm_name(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def inchikey_from_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return None


def block(inchikey: str | None) -> str | None:
    """First 14-char connectivity block of an InChIKey (ignores stereo/salt)."""
    return inchikey.split("-")[0] if isinstance(inchikey, str) and inchikey else None


def _valid_block(x) -> bool:
    """True only for a real InChIKey block string (guards against None *and* NaN)."""
    return isinstance(x, str) and len(x) > 0


# ---------------------------------------------------------------- PubChem ----
def _load_cache() -> dict:
    if PUBCHEM_CACHE.exists():
        return json.loads(PUBCHEM_CACHE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    PUBCHEM_CACHE.write_text(json.dumps(cache, indent=0))


def pubchem_name_to_inchikey(name: str, cache: dict) -> str | None:
    """Resolve a single name via PubChem; None cached as '' to avoid re-querying."""
    key = name.strip()
    if key in cache:
        return cache[key] or None
    val = None
    try:
        r = requests.get(
            f"{PUG}/compound/name/{requests.utils.quote(key)}/property/InChIKey/JSON",
            timeout=25,
        )
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"]
            if props:
                val = props[0].get("InChIKey")
    except Exception:
        val = None
    cache[key] = val or ""
    time.sleep(0.22)  # stay under PubChem's ~5 req/s limit
    return val


def _clean_name(name: str) -> str:
    """Strip parenthetical qualifiers that break PubChem lookups:
    `Nutlin-3a (-)` -> `Nutlin-3a`, `Bleomycin (50 uM)` -> `Bleomycin`."""
    s = re.sub(r"\([^)]*\)", "", str(name))
    return re.sub(r"\s+", " ", s).strip()


def resolve_gdsc_inchikey(name: str, synonyms: str, cache: dict) -> str | None:
    """Try the primary name, a parenthetical-stripped variant, then each synonym
    (and its stripped variant), until PubChem resolves one."""
    raw = [name] + ([s.strip() for s in str(synonyms).split(",")] if pd.notna(synonyms) else [])
    cands, seen = [], set()
    for c in raw:
        for v in (c, _clean_name(c)):
            v = (v or "").strip()
            if v and v.lower() not in seen:
                seen.add(v.lower())
                cands.append(v)
    for cand in cands:
        ik = pubchem_name_to_inchikey(cand, cache)
        if ik:
            return ik
    return None


# ------------------------------------------------------------- build tables ---
def build_ctrp_drugs() -> pd.DataFrame:
    cpd = pd.read_csv(CTRP_CPD, sep="\t")
    cpd = cpd.rename(
        columns={
            "cpd_name": "drug_name",
            "gene_symbol_of_protein_target": "target",
            "target_or_activity_of_compound": "target_activity",
        }
    )
    cpd["cohort"] = "CTRP"
    cpd["native_drug_id"] = cpd["index_cpd"].astype(str)
    cpd["inchikey"] = cpd["cpd_smiles"].map(inchikey_from_smiles)
    cpd["nname"] = cpd["drug_name"].map(norm_name)
    return cpd[
        ["cohort", "native_drug_id", "drug_name", "nname", "inchikey",
         "broad_cpd_id", "target"]
    ]


def build_gdsc_drugs(gdsc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if gdsc_df is None:
        gdsc_df = pd.read_excel(GDSC_XLSX)
    g = (
        gdsc_df[["DRUG_ID", "DRUG_NAME", "PUTATIVE_TARGET"]]
        .drop_duplicates("DRUG_ID")
        .rename(columns={"DRUG_ID": "native_drug_id", "DRUG_NAME": "drug_name",
                         "PUTATIVE_TARGET": "target"})
    )
    anno = pd.read_csv(GDSC_ANNO)[["DRUG_ID", "SYNONYMS"]].rename(
        columns={"DRUG_ID": "native_drug_id"}
    )
    g = g.merge(anno, on="native_drug_id", how="left")
    g["native_drug_id"] = g["native_drug_id"].astype(str)
    g["cohort"] = "GDSC"
    g["nname"] = g["drug_name"].map(norm_name)

    cache = _load_cache()
    iks = []
    for i, row in enumerate(g.itertuples(index=False), 1):
        iks.append(resolve_gdsc_inchikey(row.drug_name, row.SYNONYMS, cache))
        if i % 25 == 0:
            _save_cache(cache)
            print(f"  PubChem resolved {i}/{len(g)} GDSC drugs...")
    _save_cache(cache)
    g["inchikey"] = iks
    g["broad_cpd_id"] = pd.NA
    return g[["cohort", "native_drug_id", "drug_name", "nname", "inchikey",
              "broad_cpd_id", "target", "SYNONYMS"]]


def build_drug_crosswalk(gdsc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    ctrp = build_ctrp_drugs()
    gdsc = build_gdsc_drugs(gdsc_df)

    for df in (ctrp, gdsc):
        df["ik_block"] = df["inchikey"].map(block)

    # Cross-cohort identity is STRUCTURE ONLY: two compounds are the same drug iff
    # their InChIKey connectivity blocks match. We deliberately do NOT merge on name
    # alone -- same-name/different-structure cases are real (e.g. the bosutinib
    # isomer; GDSC UBPYILGKFZZVDX vs CTRP ALUXPRDJABOEAG) and merging them would
    # pool two different molecules. Such cases are surfaced by same_name_diff_struct().
    gdsc_blocks = {b for b in gdsc["ik_block"] if _valid_block(b)}
    ctrp_blocks = {b for b in ctrp["ik_block"] if _valid_block(b)}

    def _assign(df, other_blocks):
        uids, chans = [], []
        for row in df.itertuples(index=False):
            blk = row.ik_block
            if _valid_block(blk):
                uids.append(blk)
                chans.append("inchikey" if blk in other_blocks else "cohort_only")
            else:
                # no resolved structure -> stable name slug, COHORT-PREFIXED so two
                # structurally-unconfirmed compounds that merely share a name can never
                # collapse into one cross-cohort drug_uid (structure-only identity).
                uids.append(f"name:{row.cohort}:{row.nname}")
                chans.append("unresolved")
        return df.assign(drug_uid=uids, match_channel=chans)

    gdsc = _assign(gdsc, ctrp_blocks)
    ctrp = _assign(ctrp, gdsc_blocks)

    keep = ["cohort", "native_drug_id", "drug_name", "nname", "inchikey",
            "ik_block", "drug_uid", "match_channel", "broad_cpd_id", "target"]
    return pd.concat([gdsc[keep], ctrp[keep]], ignore_index=True, sort=False)


def same_name_diff_struct(xw: pd.DataFrame) -> pd.DataFrame:
    """Drugs whose normalized name appears in BOTH cohorts but whose InChIKey
    connectivity blocks never match -- flagged for manual review, NOT merged."""
    g = xw[xw.cohort == "GDSC"]
    c = xw[xw.cohort == "CTRP"]
    shared_names = set(g["nname"]) & set(c["nname"])
    rows = []
    for nm in sorted(shared_names):
        gb = {b for b in g.loc[g.nname == nm, "ik_block"] if _valid_block(b)}
        cb = {b for b in c.loc[c.nname == nm, "ik_block"] if _valid_block(b)}
        if gb and cb and not (gb & cb):
            rows.append({
                "nname": nm,
                "gdsc_name": g.loc[g.nname == nm, "drug_name"].iloc[0],
                "gdsc_block": ";".join(sorted(gb)),
                "ctrp_block": ";".join(sorted(cb)),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    REF.mkdir(exist_ok=True)
    xw = build_drug_crosswalk()
    out = REF / "drug_crosswalk.parquet"
    xw.to_parquet(out, index=False)
    g = xw[xw.cohort == "GDSC"]
    c = xw[xw.cohort == "CTRP"]
    shared_uid = set(g.drug_uid) & set(c.drug_uid)
    print(f"\nwrote {out}  ({len(xw)} rows)")
    print(f"GDSC drugs: {g.drug_uid.nunique()}  (from {len(g)} DRUG_IDs)  InChIKey-resolved: {g.inchikey.notna().sum()}")
    print(f"CTRP drugs: {c.drug_uid.nunique()}  (from {len(c)} cpds)  InChIKey-resolved: {c.inchikey.notna().sum()}")
    print(f"shared drug_uid (structure-confirmed): {len(shared_uid)}")
    print(f"union drug_uid: {xw.drug_uid.nunique()}")
    print(f"null drug_uid (should be 0): {xw.drug_uid.isna().sum()}")
    print("\nGDSC match_channel:\n", g.match_channel.value_counts().to_string())

    flags = same_name_diff_struct(xw)
    flags.to_csv(REF / "same_name_diff_structure.csv", index=False)
    print(f"\nsame-name / different-structure (flagged, NOT merged): {len(flags)}")
    if len(flags):
        print(flags.to_string(index=False))
