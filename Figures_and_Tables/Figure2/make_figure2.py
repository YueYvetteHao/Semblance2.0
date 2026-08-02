#!/usr/bin/env python3
"""Figure 2 — Broad-spectrum vs mechanism, the pan-cancer flip, and the tissue-free test.
Reads ./data/ (ablation_results_v1.csv, ablation_results_pancancer.csv, ablation_results_tissuefree.csv);
writes Figure2.png/.pdf. Palette validated with the dataviz skill (blue=broad, orange=mechanism)."""
import os, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
BROAD,MECH="#2a78d6","#eb6834"; MECHLT="#f6b79c"
INK,INK2,MUTED,GRID,BASELN="#0b0b0b","#52514e","#898781","#e1e0d9","#c3c2b7"
NEUTRAL="#b8b7b1"; MM=1/25.4
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans","Arial","Helvetica","Nimbus Sans","DejaVu Sans"],
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","font.size":8,"axes.titlesize":8.6,"axes.labelsize":8,
 "xtick.labelsize":7,"ytick.labelsize":7,"axes.edgecolor":BASELN,"axes.linewidth":0.7,"xtick.color":MUTED,
 "ytick.color":MUTED,"text.color":INK,"axes.labelcolor":INK2,"axes.titlecolor":INK,"axes.spines.top":False,
 "axes.spines.right":False,"legend.frameon":False,"legend.fontsize":6.4})
cd=lambda s:s.replace("COAD/READ","COAD-READ").replace("SITE_","")
abl=pd.read_csv(f"{DATA}/ablation_results_v1.csv"); ablp=pd.read_csv(f"{DATA}/ablation_results_pancancer.csv")
ablt=pd.read_csv(f"{DATA}/ablation_results_tissuefree.csv")

fig=plt.figure(figsize=(178*MM,168*MM))
gs=fig.add_gridspec(3,2,height_ratios=[1.0,0.92,1.05],width_ratios=[1.32,1.0],hspace=0.95,wspace=0.44)

# a — per-disease dumbbell
ax=fig.add_subplot(gs[:,0]); both=abl[abl.cohort=="both"]
dm=both[both.model=="drugmean"].set_index("disease"); me=both[both.model=="enet_resid"].set_index("disease")
common=dm.index.intersection(me.index); order=dm.loc[common,"spearman"].sort_values().index; y=np.arange(len(order))
for i,dis in zip(y,order): ax.plot([me.loc[dis,"spearman"],dm.loc[dis,"spearman"]],[i,i],color=GRID,lw=1.2,zorder=1)
ax.errorbar(dm.loc[order,"spearman"],y,xerr=[dm.loc[order,"spearman"]-dm.loc[order,"spearman_lo"],dm.loc[order,"spearman_hi"]-dm.loc[order,"spearman"]],
            fmt="o",ms=4.2,color=BROAD,ecolor=BROAD,elinewidth=0.8,capsize=0,label="drug-mean (broad-spectrum)",zorder=3)
ax.errorbar(me.loc[order,"spearman"],y,xerr=[me.loc[order,"spearman"]-me.loc[order,"spearman_lo"],me.loc[order,"spearman_hi"]-me.loc[order,"spearman"]],
            fmt="o",ms=4.2,color=MECH,ecolor=MECH,elinewidth=0.8,capsize=0,label="enet_resid (mechanism)",zorder=3)
ax.axvline(0,color=BASELN,lw=0.9,zorder=0); ax.set_yticks(y); ax.set_yticklabels([cd(x) for x in order],fontsize=6)
ax.set_xlabel("held-out Spearman ρ  (vs drug-mean baseline)")
ax.set_title("a",loc="left",fontweight="bold")
ax.legend(loc="lower right",bbox_to_anchor=(1.0,0.02))
ax.text(0.985,0.185,"mechanism clears CI>0\nin only 1/20 (SARC)",transform=ax.transAxes,ha="right",fontsize=6,color=INK2)

# b — the flip
axf=fig.add_subplot(gs[0,1])
pd_dm=both[both.model=="drugmean"]["spearman"].mean(); pd_me=both[both.model=="enet_resid"]["spearman"].mean()
pc=ablp[ablp.cohort=="both"].set_index("model")
pc_dm=pc.loc["drugmean","spearman"]; pc_me=pc.loc["enet_resid","spearman"]
pc_lo,pc_hi=pc.loc["enet_resid","spearman_lo"],pc.loc["enet_resid","spearman_hi"]; xs=[0,1]
axf.plot(xs,[pd_dm,pc_dm],"-o",color=BROAD,ms=5,lw=1.8,label="broad")
axf.plot(xs,[pd_me,pc_me],"-o",color=MECH,ms=5,lw=1.8,label="mechanism")
axf.errorbar([1],[pc_me],yerr=[[pc_me-pc_lo],[pc_hi-pc_me]],fmt="none",ecolor=MECH,elinewidth=1,capsize=2)
axf.axhline(0,color=BASELN,lw=0.8); axf.set_xticks(xs)
axf.set_xticklabels(["per-disease\n(mean, n=20)","pan-cancer\n(n≈398k)"],fontsize=6.2)
axf.set_xlim(-0.25,1.25); axf.set_ylim(-0.06,0.25); axf.set_ylabel("Spearman ρ")
for xi,yi in [(0,pd_dm),(1,pc_dm)]: axf.text(xi,yi+0.014,f"{yi:+.2f}",ha="center",color=BROAD,fontsize=6,fontweight="bold")
axf.text(0,pd_me-0.032,f"{pd_me:+.2f}",ha="center",color=MECH,fontsize=6,fontweight="bold")
axf.text(1,pc_me+0.016,f"{pc_me:+.2f}",ha="center",color=MECH,fontsize=6,fontweight="bold")
axf.set_title("b",loc="left",fontweight="bold"); axf.legend(loc="center left",bbox_to_anchor=(0.02,0.55))

# c — pooling narrows CI
axc2=fig.add_subplot(gs[1,1]); w=abl[abl.model=="enet_resid"].copy(); w["ciw"]=w["spearman_hi"]-w["spearman_lo"]
ciw=w.groupby("cohort")["ciw"].mean().reindex(["gdsc","ctrp","both"])
axc2.bar(range(3),ciw.values,color=[NEUTRAL,NEUTRAL,BROAD],width=0.6)
axc2.set_xticks(range(3)); axc2.set_xticklabels(["GDSC","CTRP","pooled"],fontsize=6.5)
for i,v in enumerate(ciw.values): axc2.text(i,v+0.0015,f"{v:.3f}",ha="center",fontsize=6,color=INK2)
axc2.set_ylabel("mean 95% CI width"); axc2.set_ylim(0,ciw.max()*1.30)
axc2.set_title("c",loc="left",fontweight="bold")

# d — tissue-free test: is the pan-cancer signal personalization or recovered lineage?
axd=fig.add_subplot(gs[2,1]); tf=ablt[ablt.cohort=="both"].set_index("model")
oracle=tf.loc["disease_drugmean","spearman"]; drugres=tf.loc["enet_resid","spearman"]
tfree=tf.loc["enet_resid_disease","spearman"]; pnull=tf.loc["enet_resid_disease","perm_mean"]; pp=tf.loc["enet_resid_disease","perm_p"]
names=["tissue baseline\n(no expression)","expression\n− drug mean","expression\n− tissue×drug"]
vals=[oracle,drugres,tfree]; cols=[NEUTRAL,MECHLT,MECH]; yb=np.array([2,1,0])
axd.barh(yb,vals,color=cols,height=0.6,zorder=2)
axd.axvline(0,color=BASELN,lw=0.8,zorder=0)
axd.axvline(pnull,color=MUTED,lw=0.8,ls=(0,(3,2)),zorder=1)
axd.text(pnull,2.92,"perm.\nnull",color=MUTED,fontsize=5.0,ha="center",va="bottom",linespacing=0.9)
axd.text(oracle+0.006,2,f"{oracle:+.3f}",va="center",fontsize=6,color=INK2)
axd.text(drugres+0.006,1,f"{drugres:+.3f}",va="center",fontsize=6,color=INK2)
axd.text(tfree+0.006,0,f"{tfree:+.3f}  **",va="center",fontsize=6,color=MECH,fontweight="bold")
# NOTE: the arrow between the two mechanism bars and its "43% tissue" label were removed. The two
# bars are correlations against DIFFERENT (nested) targets, so the gap between them is a drop in
# held-out skill, not a share of one signal — annotating it as a percentage read as a variance
# decomposition, which is a claim the data do not support. The caption states the comparison instead.
axd.set_yticks(yb); axd.set_yticklabels(names,fontsize=5.7,linespacing=0.95)
axd.set_xlim(0,0.265); axd.set_ylim(-0.55,3.5); axd.set_xlabel("pan-cancer held-out ρ  (≥5 lines/cell)",fontsize=6.6)
axd.set_title("d",loc="left",fontweight="bold",fontsize=8.2)
axd.text(0.985,0.045,f"tissue-free ρ survives\nwithin-tissue null (p={pp:.3f})",transform=axd.transAxes,ha="right",va="bottom",fontsize=5.6,color=INK2,linespacing=1.0)

fig.savefig(f"{HERE}/Figure2.pdf",bbox_inches="tight"); fig.savefig(f"{HERE}/Figure2.png",dpi=400,bbox_inches="tight",facecolor="white")
print("wrote Figure2  (oracle %.3f | -drug %.3f | -tissue %.3f | perm_p %.4f)"%(oracle,drugres,tfree,pp))
