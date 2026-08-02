#!/usr/bin/env python3
"""Figure (novel signals) — permutation-validated personalization beyond canonical biomarkers.
Reads ./data/ (novel_hits.csv, summary.csv, melanoma_gpx4.csv); writes Figure4.png/.pdf.
(a) mechanism-grouped novel recurrent hits; (b) recurrence spectrum; (c) melanoma ferroptosis case."""
import os, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
INK,INK2,MUTED,GRID,BASELN="#0b0b0b","#52514e","#898781","#e1e0d9","#c3c2b7"; MM=1/25.4
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans","Arial","Helvetica","Nimbus Sans","DejaVu Sans"],
 "mathtext.fontset":"dejavusans","mathtext.default":"regular",
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","font.size":8,"axes.titlesize":8.6,"axes.labelsize":8,
 "xtick.labelsize":7,"ytick.labelsize":7,"axes.edgecolor":BASELN,"axes.linewidth":0.7,"xtick.color":MUTED,
 "ytick.color":MUTED,"text.color":INK,"axes.labelcolor":INK2,"axes.titlecolor":INK,"axes.spines.top":False,
 "axes.spines.right":False,"legend.frameon":False,"legend.fontsize":6.2})

cur=pd.read_csv(f"{DATA}/novel_hits.csv")
summ=pd.read_csv(f"{DATA}/summary.csv",index_col=0).iloc[:,0]
mel=pd.read_csv(f"{DATA}/melanoma_gpx4.csv")

# mechanism-targeted groups first (headline), broadly-cytotoxic last; short display labels
GROUPS=["Ferroptosis (GPX4)","BH3-mimetic (BCL2 fam.)","Nuclear export (XPO1)",
        "Mitotic checkpoint","Proteostasis / survivin","Topoisomerase / DNA-damage","Antimetabolite"]
SRCLABEL={"Mitotic checkpoint":"Mitotic checkpoint (WEE1/AURK/PLK1)"}  # data label -> display
DATAMECH={g:SRCLABEL.get(g,g) for g in GROUPS}
COL={"Ferroptosis (GPX4)":"#eb6834","BH3-mimetic (BCL2 fam.)":"#c13d86","Nuclear export (XPO1)":"#6a52c9",
     "Mitotic checkpoint":"#2b8fb0","Proteostasis / survivin":"#2f9e6f",
     "Topoisomerase / DNA-damage":"#9a8352","Antimetabolite":"#a7a6a0"}
CYTO={"Topoisomerase / DNA-damage","Antimetabolite"}

fig=plt.figure(figsize=(206*MM,152*MM))
gs=fig.add_gridspec(2,2,height_ratios=[1.0,1.0],width_ratios=[1.6,1.0],hspace=0.5,wspace=0.22)

# ---------- a: mechanism-grouped lollipop (targeted at TOP) ----------
ax=fig.add_subplot(gs[:,0])
y=0.0; gap=1.5; placed=[]; gspan={}
for g in reversed(GROUPS):                       # build bottom->top so GROUPS[0] ends at top
    sub=cur[cur.mechanism==DATAMECH[g]].sort_values("best_rho")   # ascending -> highest at group top
    y0=y
    for _,r in sub.iterrows():
        placed.append((y,r,g)); y+=1.0
    gspan[g]=(y0,y-1.0); y+=gap
ytop=y-gap
band=[gspan[g] for g in GROUPS if g in CYTO]
if band:
    lo=min(b[0] for b in band)-0.6; hi=max(b[1] for b in band)+0.6
    ax.axhspan(lo,hi,color="#f4f3ef",zorder=0)
    ax.text(0.80,(lo+hi)/2,"broadly cytotoxic\n(may track general\nchemosensitivity)",rotation=90,
            va="center",ha="center",fontsize=5.2,color=MUTED,linespacing=1.0)
yt=[]; ylab=[]
for yy,r,g in placed:
    c=COL[g]
    ax.plot([0,r["best_rho"]],[yy,yy],color=c,lw=1.0,alpha=0.45,zorder=1)
    ax.scatter(r["best_rho"],yy,s=16+r["n_tissues"]*13,color=c,zorder=3,edgecolor="white",linewidth=0.5)
    ax.text(r["best_rho"]+0.013,yy,f"{int(r['n_tissues'])}",va="center",ha="left",fontsize=5.3,color=INK2)
    yt.append(yy); ylab.append(r["drug"])
# group labels in the gap above each group
for g in GROUPS:
    top=gspan[g][1]
    ax.text(0.0,top+0.62,g,va="bottom",ha="left",fontsize=6.2,fontweight="bold",color=COL[g])
ax.set_yticks(yt); ax.set_yticklabels(ylab,fontsize=6.1)
for t,(_,_,g) in zip(ax.get_yticklabels(),placed):
    if g=="Ferroptosis (GPX4)": t.set_fontweight("bold")
ax.set_ylim(-0.8,ytop+1.4); ax.set_xlim(0,0.86)
ax.set_xlabel("best held-out signature ρ  (permutation-validated, p < 0.05)")
ax.set_title("a",loc="left",fontweight="bold",fontsize=8.4)
ax.text(0.0,ytop+1.28,"number by each point = tissues the signature validated in  (larger point = more)",
        fontsize=5.4,color=INK2,va="top")

# ---------- b: recurrence spectrum ----------
axb=fig.add_subplot(gs[0,1])
steps=[("≥1 tissue (all novel)",int(summ["novel_drugs"])),("≥2 tissues",int(summ["novel_ge2"])),
       ("≥4 tissues",int(summ["novel_ge4"])),("≥6 tissues",int(summ["novel_ge6"]))]
labs=[s[0] for s in steps]; vals=[s[1] for s in steps]; yy=np.arange(len(steps))[::-1]
axb.barh(yy,vals,color=["#c3c2b7","#8fb8e0","#4f90d9","#2a78d6"],height=0.66)
for yi,vv in zip(yy,vals): axb.text(vv+7,yi,f"{vv}",va="center",fontsize=6.6,fontweight="bold",color=INK)
axb.set_yticks(yy); axb.set_yticklabels(labs,fontsize=6.1); axb.set_xlim(0,480)
axb.set_xlabel("distinct novel drugs",fontsize=7)
axb.set_title("b",loc="left",fontweight="bold",fontsize=8.4)
axb.text(0.985,0.09,"recurrence in 4 or more tissues exceeds a\npermutation null (p < 0.001); recurrence in\n2 tissues alone does not",transform=axb.transAxes,
         ha="right",va="bottom",fontsize=5.5,color=INK2)

# ---------- c: melanoma ferroptosis case ----------
axc=fig.add_subplot(gs[1,1])
m=mel.copy(); m["lab"]=m["drug_name"].replace({"1S,3R-RSL-3":"RSL3"})
m=m.sort_values("spearman"); yy=np.arange(len(m))
axc.barh(yy,m["spearman"],color="#eb6834",height=0.58)
for yi,rho in zip(yy,m["spearman"]):
    axc.text(rho-0.02,yi,f"{rho:.2f}",va="center",ha="right",fontsize=6.4,color="white",fontweight="bold")
axc.set_yticks(yy); axc.set_yticklabels(m["lab"],fontsize=6.6); axc.set_xlim(0,0.8); axc.set_ylim(-0.9,len(m)-0.35)
axc.set_xlabel("held-out ρ in melanoma (SKCM)",fontsize=7)
axc.set_title("c",loc="left",fontweight="bold",fontsize=8.4)
axc.text(0.0,-0.82,"3 independent GPX4 inhibitors, all validated in SKCM (p = 0.010)",
         ha="left",va="center",fontsize=5.2,color=INK2)

fig.savefig(f"{HERE}/Figure4.pdf",bbox_inches="tight"); fig.savefig(f"{HERE}/Figure4.png",dpi=400,bbox_inches="tight",facecolor="white")
print("wrote Figure4  | novel hits %d, novel drugs %d, recur>=2 %d"%(summ["novel_hits"],summ["novel_drugs"],summ["novel_ge2"]))
