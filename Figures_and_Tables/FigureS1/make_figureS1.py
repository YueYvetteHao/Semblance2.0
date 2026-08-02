#!/usr/bin/env python3
"""Figure S1 — Harmonized resource + QC (4 panels). Self-contained: reads ./data/, writes FigureS1.png/.pdf.
Palette validated with the dataviz skill."""
import os, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
BROAD,MECH="#2a78d6","#eb6834"; INK,INK2,MUTED,GRID,BASELN="#0b0b0b","#52514e","#898781","#e1e0d9","#c3c2b7"
BLUES=LinearSegmentedColormap.from_list("b",["#eef4fd","#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"])
MM=1/25.4
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans","Arial","Helvetica","Nimbus Sans","DejaVu Sans"],
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","font.size":8,"axes.titlesize":9,"axes.labelsize":8,
 "xtick.labelsize":7,"ytick.labelsize":7,"axes.edgecolor":BASELN,"axes.linewidth":0.7,"xtick.color":MUTED,
 "ytick.color":MUTED,"text.color":INK,"axes.labelcolor":INK2,"axes.titlecolor":INK,"axes.spines.top":False,
 "axes.spines.right":False,"legend.frameon":False,"legend.fontsize":7,"mathtext.fontset":"dejavusans","mathtext.default":"regular"})
cd=lambda s:s.replace("COAD/READ","COAD-READ").replace("SITE_","")

cnt=pd.read_csv(f"{DATA}/harmonization_counts.csv")
rho=pd.read_csv(f"{DATA}/concordance_per_drug_rho.csv")["spearman_rho"].values
summ=pd.read_csv(f"{DATA}/cell_lines_per_disease.csv")
cov=pd.read_csv(f"{DATA}/omics_coverage.csv")

fig=plt.figure(figsize=(174*MM,150*MM)); gs=fig.add_gridspec(2,2,hspace=0.48,wspace=0.34)

# a — counts
axa=fig.add_subplot(gs[0,0]); x=np.arange(4); w=0.38
axa.bar(x-w/2,cnt["cell_lines"],w,color=BROAD,label="Cell lines")
axa.bar(x+w/2,cnt["drugs"],w,color=MECH,label="Drugs")
for xi,c,d in zip(x,cnt["cell_lines"],cnt["drugs"]):
    axa.text(xi-w/2,c+15,f"{c:,}",ha="center",va="bottom",fontsize=6.2,color=INK2)
    axa.text(xi+w/2,d+15,f"{d:,}",ha="center",va="bottom",fontsize=6.2,color=INK2)
axa.set_xticks(x); axa.set_xticklabels(cnt["set"]); axa.set_ylabel("count"); axa.set_ylim(0,1330)
axa.legend(loc="upper right",bbox_to_anchor=(1.02,1.03))
axa.set_title("a",loc="left",fontweight="bold")
# The structure-vs-name compound count (71 -> 86) is stated in the caption instead of
# annotated on the panel, so the figure carries no callout box or arrow.

# b — concordance
axb=fig.add_subplot(gs[0,1]); med=np.median(rho)
axb.hist(rho,bins=22,color=BROAD,alpha=0.85,edgecolor="white",linewidth=0.4)
axb.axvline(0,color=BASELN,lw=0.8); axb.axvline(med,color=MECH,lw=1.6,ls="--")
axb.text(med+0.02,axb.get_ylim()[1]*0.93,f"median ρ = {med:.3f}",color=MECH,fontsize=6.6,fontweight="bold")
axb.text(0.03,0.80,f"{(rho>0).mean()*100:.0f}% positive",transform=axb.transAxes,fontsize=6.4,color=INK2)
axb.set_xlabel("per-drug Spearman ρ  (GDSC vs CTRP sens_z)"); axb.set_ylabel(f"drugs (n={len(rho)})")
axb.set_title("b",loc="left",fontweight="bold")

# c — cell lines per disease (FIX: short title)
axc=fig.add_subplot(gs[1,0]); s20=summ.sort_values("n_cells",ascending=False).head(20).copy()
s20["disease"]=s20["disease"].map(cd); yy=np.arange(len(s20))[::-1]
axc.barh(yy,s20["n_cells"],color=BROAD,height=0.72)
axc.set_yticks(yy); axc.set_yticklabels(s20["disease"],fontsize=5.8)
for y,v in zip(yy,s20["n_cells"]): axc.text(v+0.6,y,str(int(v)),va="center",fontsize=5.6,color=INK2)
axc.set_xlabel("cell lines (expression + ≥1 drug)"); axc.set_xlim(0,78)
axc.set_title("c",loc="left",fontweight="bold")
axc.text(0.97,0.06,"vs 57,820 genes  (p $\\gg$ n)",transform=axc.transAxes,ha="right",fontsize=6,color=INK2,style="italic")

# d — omics coverage (FIX: title stays within column)
axd=fig.add_subplot(gs[1,1]); layers=["mutations","fusions","methylation","metabolomics","rppa","mirna","chromatin","cn_summary"]
cv=cov.sort_values("n_cells",ascending=False).head(20).copy()
frac=cv[layers].values/cv["n_cells"].values[:,None]
im=axd.imshow(frac,aspect="auto",cmap=BLUES,vmin=0,vmax=1)
axd.set_xticks(range(len(layers))); axd.set_xticklabels(layers,rotation=45,ha="right",fontsize=5.8)
axd.set_yticks(range(len(cv))); axd.set_yticklabels([cd(x) for x in cv["disease"]],fontsize=5.5)
cb=fig.colorbar(im,ax=axd,fraction=0.046,pad=0.03); cb.set_label("fraction profiled",fontsize=6.2); cb.ax.tick_params(labelsize=5.5)
axd.set_title("d",loc="left",fontweight="bold")

fig.savefig(f"{HERE}/FigureS1.pdf",bbox_inches="tight"); fig.savefig(f"{HERE}/FigureS1.png",dpi=400,bbox_inches="tight",facecolor="white")
print("wrote FigureS1  (concordance median %.3f)"%med)
