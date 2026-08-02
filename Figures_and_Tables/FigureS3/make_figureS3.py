#!/usr/bin/env python3
"""Figure S3 (supplementary) — Drug-mean ranking quality (normalized discounted cumulative gain over the
top 3 drugs): within-tissue works, pan-cancer is near chance.
Reads ./data/ablation_results_v1.csv + ablation_results_pancancer.csv; writes FigureS3.png/.pdf."""
import os, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
BROAD,MECH="#2a78d6","#eb6834"; INK,INK2,MUTED,BASELN="#0b0b0b","#52514e","#898781","#c3c2b7"; MM=1/25.4
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans","Arial","Helvetica","Nimbus Sans","DejaVu Sans"],
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","font.size":8,"axes.labelsize":8,"xtick.labelsize":7,
 "ytick.labelsize":7,"axes.edgecolor":BASELN,"axes.linewidth":0.7,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK2,"axes.spines.top":False,"axes.spines.right":False})
cd=lambda s:s.replace("COAD/READ","COAD-READ").replace("SITE_","")
abl=pd.read_csv(f"{DATA}/ablation_results_v1.csv"); ablp=pd.read_csv(f"{DATA}/ablation_results_pancancer.csv")
r=abl[(abl.cohort=="both")&(abl.model=="drugmean")].dropna(subset=["rank_ndcg@3"]).copy()
r["disease"]=r["disease"].map(cd); r=r.sort_values("rank_ndcg@3"); yy=np.arange(len(r))
fig,ax=plt.subplots(figsize=(120*MM,82*MM))
ax.barh(yy,r["rank_ndcg@3"],color=BROAD,height=0.7)
pc=ablp[(ablp.cohort=="both")&(ablp.model=="drugmean")]["rank_ndcg@3"].iloc[0]
ax.axvline(pc,color=MECH,lw=1.4,ls="--"); ax.text(pc+0.007,len(r)-0.42,f"pan-cancer = {pc:.2f}",color=MECH,fontsize=6,fontweight="bold",va="bottom",ha="left")
ax.set_yticks(yy); ax.set_yticklabels(r["disease"],fontsize=5.6); ax.set_xlabel("drug-mean ranking quality\n(normalized discounted cumulative gain, top 3 drugs)")
ax.set_ylim(-0.7,len(r)+0.15)
ax.set_title("",loc="left",fontweight="bold",fontsize=8,pad=8)
fig.savefig(f"{HERE}/FigureS3.pdf",bbox_inches="tight"); fig.savefig(f"{HERE}/FigureS3.png",dpi=400,bbox_inches="tight",facecolor="white")
print("wrote FigureS3")
