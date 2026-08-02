#!/usr/bin/env python3
"""Supplementary Fig S2 — Pathway re-representation is a wash (full permutation-controlled ablation).
15 disease panels, cohort=both. Raw ~1,800-feature Reactome pathway arm (enet_pathway) held-out Spearman,
coloured by permutation significance; gene-PC mechanism arm (enet_resid) shown as reference diamonds."""
import os, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
INK,INK2,MUTED,BASELN="#0b0b0b","#52514e","#898781","#c3c2b7"
SIG,NS,GENE="#0ca30c","#c9c8c1","#eb6834"; MM=1/25.4
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans","Arial","Helvetica","Nimbus Sans","Liberation Sans","DejaVu Sans"],
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","font.size":8,"axes.labelsize":8,"xtick.labelsize":7,
 "ytick.labelsize":7,"axes.edgecolor":BASELN,"axes.linewidth":0.7,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK2,"axes.spines.top":False,"axes.spines.right":False})
a=pd.read_csv(f"{DATA}/pathway_full.csv"); b=a[a.cohort=="both"]
pw=b[b.model=="enet_pathway"].set_index("disease"); ge=b[b.model=="enet_resid"].set_index("disease")
order=pw["spearman"].sort_values().index; y=np.arange(len(order))
cd=lambda s:s.replace("COAD/READ","COAD-READ")
cols=[SIG if pw.loc[d,"perm_p"]<0.05 else NS for d in order]
fig,ax=plt.subplots(figsize=(120*MM,88*MM))
ax.barh(y,pw.loc[order,"spearman"],color=cols,height=0.66,zorder=2)
ax.scatter(ge.loc[order,"spearman"],y,marker="D",s=17,color=GENE,zorder=3,label="gene-PC mechanism arm")
ax.axvline(0,color=BASELN,lw=0.9)
ax.set_yticks(y); ax.set_yticklabels([cd(d) for d in order],fontsize=6.4)
ax.set_xlabel("held-out Spearman ρ  (pathway arm vs drug-mean-residualized target)")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
leg=[Patch(fc=SIG,label="pathway arm, perm p<0.05"),Patch(fc=NS,label="pathway arm, n.s."),
     Line2D([0],[0],marker="D",color="w",markerfacecolor=GENE,markersize=5,label="gene-PC mechanism arm")]
ax.legend(handles=leg,loc="lower right",fontsize=5.8,frameon=False)
fig.savefig(f"{HERE}/FigureS2.pdf",bbox_inches="tight"); fig.savefig(f"{HERE}/FigureS2.png",dpi=400,bbox_inches="tight",facecolor="white")
print("wrote FigureS2")
