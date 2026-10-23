#!/usr/bin/env python3
"""Reproduce Fig. 10 data/plot from the archived SCA timing records and grid median."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'sca_100_timings.csv').open()))
grid=json.loads((ROOT/'grid_serial20_summary.json').read_text())
offsets=[-20,-10,0,10,20];areas=[792,792,797,761,792]
A=float(grid['best_area_mm2']);T=float(grid['wall_seconds']['median'])
fa_all=[];rt_all=[]
for o,a in zip(offsets,areas):
    t=np.array([float(r['seconds']) for r in rows if int(r['offset_percent'])==o])
    assert len(t)==20
    fa_all.append(np.full(20,a/A));rt_all.append((T-t)/T*100)

blue,orange='#0072BD','#D95319'
fig,ax=plt.subplots(figsize=(7.4,4.7));ax2=ax.twinx();pos=np.array(offsets,float)
for axis,vals,color in [(ax,fa_all,blue),(ax2,rt_all,orange)]:
    axis.boxplot(vals,positions=pos,widths=1.65,whis=1.5,patch_artist=True,manage_ticks=False,
        boxprops=dict(facecolor=color+'25',edgecolor=color,linewidth=1.35),
        medianprops=dict(color=color,linewidth=2),whiskerprops=dict(color=color,linewidth=1.15),
        capprops=dict(color=color,linewidth=1.15),
        flierprops=dict(marker='o',markersize=3.4,markerfacecolor='white',markeredgecolor=color,markeredgewidth=1))
ax.set_xlim(-25,25);ax.set_xticks(offsets,['−20','−10','0','+10','+20'])
ax.set_xlabel('Initialization offset (%)',labelpad=9)
ax.set_ylim(.1,1.5);ax.set_yticks(np.arange(.1,1.51,.2))
ax2.set_ylim(80,120);ax2.set_yticks(np.arange(80,121,10))
ax.set_ylabel(r'Area ratio, $F_A$',color=blue,labelpad=10);ax2.set_ylabel(r'Computation-time reduction, $R_T$ (%)',color=orange,labelpad=10)
ax.tick_params(axis='y',colors=blue);ax2.tick_params(axis='y',colors=orange);ax.tick_params(direction='in');ax2.tick_params(direction='in')
ax.spines['left'].set_color(blue);ax2.spines['right'].set_color(orange);ax.spines['right'].set_visible(False);ax2.spines['left'].set_visible(False)
ax.set_axisbelow(True);ax.yaxis.grid(True,color='#dddddd',linestyle='--',linewidth=.6)
for i,o in enumerate(offsets):
    ax.annotate(f'{np.median(fa_all[i]):.3f}',(o,np.median(fa_all[i])),xytext=(0,-7),textcoords='offset points',ha='center',va='top',color=blue,fontsize=9)
    ax2.annotate(f'{np.median(rt_all[i]):.2f}%',(o,min(rt_all[i])),xytext=(0,-15),textcoords='offset points',ha='center',va='top',color=orange,fontsize=9)
ax.legend(handles=[Patch(facecolor=blue+'25',edgecolor=blue,label='Area ratio'),Patch(facecolor=orange+'25',edgecolor=orange,label='Computation-time reduction')],loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False,fontsize=10)
fig.subplots_adjust(left=.13,right=.865,bottom=.16,top=.86)
fig.savefig(ROOT/'Fig10_reproduced.png',dpi=600,bbox_inches='tight');fig.savefig(ROOT/'Fig10_reproduced.pdf',dpi=600,bbox_inches='tight')
print('grid median:',T)
for o,fa,rt in zip(offsets,fa_all,rt_all):print(o,np.median(fa),np.median(rt))
