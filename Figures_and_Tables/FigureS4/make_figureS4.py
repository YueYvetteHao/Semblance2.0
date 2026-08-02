#!/usr/bin/env python3
"""Supplementary pathway-atlas figure — sensitivity-associated Reactome programs for two validated hits.
(a) PLX-4720 (BRAF i) in melanoma — a lineage-confounded signature; (b) linsitinib (IGF1R i) in sarcoma —
a DNA-replication / cell-cycle program. Reads ./data/*_GSEA_REACTOME.csv (harmonized fgsea atlas; ranked on
-sens_z so positive NES = enriched in SENSITIVE lines). Writes FigureS4.png/.pdf."""
import os, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
INK,INK2,MUTED,BASELN="#0b0b0b","#52514e","#898781","#c3c2b7"
BLUES=LinearSegmentedColormap.from_list("b",["#eef4fd","#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]); MM=1/25.4
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans","Arial","Helvetica","Nimbus Sans","DejaVu Sans"],
 "mathtext.fontset":"dejavusans","mathtext.default":"regular",
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","font.size":8,"axes.labelsize":8,"xtick.labelsize":7,
 "ytick.labelsize":7,"axes.edgecolor":BASELN,"axes.linewidth":0.7,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK2,"axes.spines.top":False,"axes.spines.right":False})

def nice(s):
    s=s.replace("REACTOME_","").replace("_"," ").title()
    return (s[:50]+"…") if len(s)>51 else s

def panel(ax,csv,title,note,xlabel=False,topn=11):
    g=pd.read_csv(f"{DATA}/{csv}")
    sig=g[(g["p.adjust"]<0.05)&(g["NES"]>0)].sort_values("NES",ascending=False).head(topn).copy()
    sig["lab"]=sig["Description"].map(nice); yy=np.arange(len(sig))[::-1]
    norm=(sig["NES"]-sig["NES"].min())/(sig["NES"].max()-sig["NES"].min()+1e-9)
    ax.barh(yy,sig["NES"],color=BLUES(0.35+0.6*norm.values),height=0.72)
    ax.set_yticks(yy); ax.set_yticklabels(sig["lab"],fontsize=6)
    for y,v in zip(yy,sig["NES"]): ax.text(v+0.02,y,f"{v:.2f}",va="center",fontsize=5.6,color=INK2)
    ax.set_xlim(0,sig["NES"].max()*1.16); ax.set_ylim(-1.1,len(sig)-0.35)
    ax.set_title(title,loc="left",fontweight="bold",fontsize=8.1)
    ax.text(sig["NES"].max()*1.15,-0.85,note,ha="right",va="center",fontsize=6,color=INK2,style="italic")
    if xlabel: ax.set_xlabel("fgsea NES  (positive = enriched in sensitive lines)")

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(158*MM,150*MM)); fig.subplots_adjust(hspace=0.42)
panel(ax1,"PLX4720_SKCM_GSEA_REACTOME.csv",
      "a",
      "melanocytic / neural-crest lineage signature")
panel(ax2,"linsitinib_SARC_GSEA_REACTOME.csv",
      "b",
      "DNA-replication / cell-cycle program",xlabel=True)

fig.savefig(f"{HERE}/FigureS4.pdf",bbox_inches="tight"); fig.savefig(f"{HERE}/FigureS4.png",dpi=400,bbox_inches="tight",facecolor="white")
print("wrote FigureS4 (2 panels: PLX-4720/SKCM + linsitinib/SARC)")
