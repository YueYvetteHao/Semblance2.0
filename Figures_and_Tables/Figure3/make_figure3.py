#!/usr/bin/env python3
"""Figure 3 — Per-drug signatures: real biomarkers, and why the CV interval overcounts.
Full permutation-controlled F2 sweep (7,836 disease×drug pairs, 20 diseases).
(a) significance funnel 7836 -> 3284 (naive CI) -> 971 (permutation);
(b) permutation null vs observed signature rho (why CI_lo>0 overcounts);
(c) validated-hit forest, coloured by target class (known biology recovered)."""
import os, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
INK,INK2,MUTED,BASELN="#0b0b0b","#52514e","#898781","#c3c2b7"
GOOD,WARN,NEUTRAL,CRIT="#0ca30c","#fab219","#b8b7b1","#d03b3b"; MM=1/25.4
NULLC,OBSC="#b9b7ae","#2a78d6"
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans","Arial","Helvetica","Nimbus Sans","Liberation Sans","DejaVu Sans"],
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","font.size":8,"axes.labelsize":8,"xtick.labelsize":7,
 "ytick.labelsize":7,"axes.edgecolor":BASELN,"axes.linewidth":0.7,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK2,"axes.titlecolor":INK,"axes.spines.top":False,"axes.spines.right":False})

fig=plt.figure(figsize=(174*MM,150*MM))
gs=fig.add_gridspec(2,2,height_ratios=[1.0,1.28],width_ratios=[1.0,1.0],hspace=0.52,wspace=0.34)

# ---------- a: funnel ----------
f=pd.read_csv(f"{DATA}/funnel_counts.csv").set_index("stage")["value"]
ax=fig.add_subplot(gs[0,0])
stages=["Candidate\n(disease, drug) pairs","'Significant' by\nnaive CI lower bound > 0","Permutation-validated\n(p < 0.05)"]
vals=[int(f["Candidate pairs"]),int(f["Significant by naive CI_lo>0"]),int(f["Permutation-validated p<0.05"])]
chance=int(f["Expected by chance"]); cols=[NEUTRAL,WARN,GOOD]; y=np.arange(3)[::-1]
ax.barh(y,vals,color=cols,height=0.62)
for yi,v in zip(y,vals): ax.text(v+90,yi,f"{v:,}",va="center",fontweight="bold",fontsize=8.5,color=INK)
ax.axvline(chance,color=CRIT,lw=1.1,ls=":"); ax.text(470,-0.62,f"~{chance:,} expected by chance",color="#b03636",fontsize=5.8,ha="left",va="center")
ax.set_yticks(y); ax.set_yticklabels(stages,fontsize=6.4); ax.set_xlabel("number of (disease, drug) panels"); ax.set_xlim(0,9200); ax.set_ylim(-1.05,2.55)
ax.set_title("a",loc="left",fontweight="bold",fontsize=8.3)

# ---------- b: permutation null vs observed ----------
axb=fig.add_subplot(gs[0,1])
s=pd.read_csv(f"{DATA}/f2_signature_all.csv")
rng=np.random.default_rng(7)
# reconstruct pooled permutation null from per-pair (mean, sd) moments
pm=s.dropna(subset=["perm_mean","perm_sd"])
null=rng.normal(pm["perm_mean"].values, pm["perm_sd"].clip(lower=1e-3).values)
bins=np.linspace(-0.9,0.9,60)
axb.hist(null,bins=bins,color=NULLC,alpha=0.85,label="permutation null\n(shuffled labels)",density=True)
axb.hist(s["spearman"].dropna(),bins=bins,histtype="step",color=OBSC,lw=1.6,label="observed signatures",density=True)
axb.axvline(0,color=BASELN,lw=0.8)
# 95th percentile of null = permutation significance bar
p95=np.quantile(null,0.95)
axb.axvline(p95,color=CRIT,lw=1.0,ls="--")
axb.text(p95+0.02,axb.get_ylim()[1]*0.92,"≈95th pct\nof null",color="#b03636",fontsize=5.6,va="top")
axb.set_xlabel("held-out signature Spearman ρ"); axb.set_ylabel("density"); axb.set_xlim(-0.9,0.9)
axb.legend(loc="upper left",fontsize=5.8,frameon=False,handlelength=1.2)
axb.set_title("b",loc="left",fontweight="bold",fontsize=8.3)

# ---------- c: validated-hit forest ----------
axc=fig.add_subplot(gs[1,:])
h=pd.read_csv(f"{DATA}/validated_hits.csv").sort_values("rho").reset_index(drop=True)
classes=["BRAF","MEK","EGFR/HER2","IGF1R","MDM2/p53","PLK1","mTOR","CHK1","PARP","BCL2","SRC/ABL","HDAC"]
palette={"BRAF":"#d94f3d","MEK":"#e07b39","EGFR/HER2":"#c99a1f","IGF1R":"#6f9e34","MDM2/p53":"#2f9e6f",
 "PLK1":"#2b8fb0","mTOR":"#3667c4","CHK1":"#6a52c9","PARP":"#a648b0","BCL2":"#c13d86","SRC/ABL":"#8a6d3b","HDAC":"#5b6770"}
yy=np.arange(len(h))
for i,r in h.iterrows():
    axc.plot([0,r["rho"]],[i,i],color=palette.get(r["tclass"],MUTED),lw=1.0,alpha=0.5,zorder=1)
    axc.scatter(r["rho"],i,color=palette.get(r["tclass"],MUTED),s=34,zorder=3,edgecolor="white",linewidth=0.5)
labs=[f'{r.drug} → {r.disease.replace("COAD/READ","COAD-READ")}' for _,r in h.iterrows()]
axc.set_yticks(yy); axc.set_yticklabels(labs,fontsize=6.2)
axc.set_ylim(-0.7,len(h)-0.3); axc.axvline(0,color=BASELN,lw=0.8)
axc.set_xlabel("held-out signature Spearman ρ  (permutation-validated, p < 0.05)"); axc.set_xlim(0,0.82)
# legend by class
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=palette[c],markersize=5,label=c) for c in classes]
axc.legend(handles=handles,loc="lower right",fontsize=5.6,frameon=False,ncol=2,handletextpad=0.3,columnspacing=0.9,title="target class",title_fontsize=6)
axc.set_title("c",loc="left",fontweight="bold",fontsize=8.3)

fig.savefig(f"{HERE}/Figure3.pdf",bbox_inches="tight")
fig.savefig(f"{HERE}/Figure3.png",dpi=400,bbox_inches="tight",facecolor="white")
print("wrote Figure3  funnel=%s  hits=%d"%(vals,len(h)))
