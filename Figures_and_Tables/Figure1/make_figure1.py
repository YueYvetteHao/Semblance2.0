#!/usr/bin/env python3
"""Figure 1 — Study & pipeline overview (schematic; no underlying data).
Self-contained: `python3 make_figure1.py` writes Figure1.png (400 dpi) + Figure1.pdf (vector).
Palette validated with the dataviz skill: blue=broad #2a78d6, orange=mechanism #eb6834."""
import os, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE=os.path.dirname(os.path.abspath(__file__))
BROAD,MECH,VIOLET="#2a78d6","#eb6834","#4a3aa7"
INK,INK2,MUTED,BASELN="#0b0b0b","#52514e","#898781","#c3c2b7"
GOOD="#0ca30c"; MM=1/25.4
mpl.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["Liberation Sans","Arial","Helvetica","Nimbus Sans","DejaVu Sans"],
    "mathtext.fontset":"dejavusans","mathtext.default":"regular",
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

fig,ax=plt.subplots(figsize=(174*MM,150*MM)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

def box(x,y,w,h,title,sub,fc,ec,tc=INK,fs=8.4,sfs=6.6):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle="round,pad=0.004,rounding_size=0.018",fc=fc,ec=ec,lw=1.2))
    ax.text(x,y+(h*0.17 if sub else 0),title,ha="center",va="center",fontsize=fs,fontweight="bold",color=tc)
    if sub: ax.text(x,y-h*0.20,sub,ha="center",va="center",fontsize=sfs,color=INK2)

def arrow(x1,y1,x2,y2,c=MUTED,lw=1.5):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle="-|>",color=c,lw=lw,shrinkA=1,shrinkB=1))

# Stage 1 — inputs
box(0.19,0.875,0.30,0.085,"GDSC2 · Sanger","967 lines · 285 drugs","#eaf2fc",BROAD)
box(0.50,0.875,0.30,0.085,"CTRP v2.2 · Broad","645 lines · 479 cpds","#fdeee6",MECH)
box(0.81,0.875,0.30,0.085,"CCLE · Broad","57,820 genes · 8 omics","#efedf8",VIOLET)
for x in (0.19,0.50,0.81): arrow(x,0.8325,x if x==0.5 else 0.5,0.775 if x==0.5 else 0.782)

# Stage 2 — harmonize
box(0.5,0.735,0.86,0.085,"Harmonize","cell → DepMap ID   ·   drug → InChIKey (structure)   ·   metric = sens_z  (− = sensitive)","#f4f4f2","#5b5a56")
arrow(0.5,0.6925,0.5,0.655)
# Stage 3 — universe
box(0.5,0.615,0.86,0.085,"Model-ready dataset","810 cell lines · 37 diseases · 678 drugs · log1p TPM + 8 multi-omic layers","#f4f4f2","#5b5a56")
arrow(0.5,0.5725,0.5,0.535)

# Stage 4 — evaluation header + two arms
box(0.5,0.492,0.86,0.070,"Evaluation","grouped leave-cell-line-out nested CV  ·  scored vs drug-mean and tissue baselines","#eef1f4","#5b5a56",fs=8.0)
arrow(0.32,0.457,0.30,0.408); arrow(0.68,0.457,0.70,0.408)
box(0.285,0.365,0.34,0.085,"Broad-spectrum","drug-mean ranking\n(within-tissue; high-confidence)","#eaf2fc",BROAD,fs=8.0)
box(0.715,0.365,0.34,0.085,"Mechanism-specific","tissue-free: disease$\\times$drug residualized\n(permutation-gated; vs tissue baseline)","#fdeee6",MECH,fs=7.4)
arrow(0.285,0.3225,0.42,0.285); arrow(0.715,0.3225,0.58,0.285)

# Stage 5 — pathway
box(0.5,0.245,0.86,0.080,"Pathway analysis","ssGSEA pathway activity  ·  per-drug in-fold signatures + permutation null  ·  GSEA atlas","#f4f4f2","#5b5a56",fs=8.0)
arrow(0.5,0.205,0.5,0.163)

# Stage 6 — output
box(0.5,0.118,0.86,0.090,"Tiered drug recommendation",
    "broad-spectrum (high-confidence, within-tissue)  +  mechanism-specific (permutation-validated minority)",
    "#e9f6e9",GOOD,fs=8.6)


fig.savefig(os.path.join(HERE,"Figure1.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(HERE,"Figure1.png"),dpi=400,bbox_inches="tight",facecolor="white")
print("wrote Figure1")
