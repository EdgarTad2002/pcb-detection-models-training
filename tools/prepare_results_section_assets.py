#!/usr/bin/env python3
"""Regenerate tables/figures for the LaTeX section; preserve edited prose."""
from pathlib import Path
import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/overleaf_results_section'
FIG=ROOT/'paper/figures/meeting_20261106'

def main():
    data=json.loads((FIG/'figure_manifest.json').read_text())
    native=json.loads((ROOT/'results/champion_audit_v1_20261030/summary.json').read_text())
    primary={r['model']:r for r in data['primary']}
    for name in ['tables','figures','sections']:(OUT/name).mkdir(parents=True,exist_ok=True)
    labels=['YOLOv12s','Crop-trained YOLO26s','AGWD + Squared-Dice','PhysOT-YOLO26s',
            'WBF, two models','WBF, four models']
    rows=[]
    for label,r in zip(labels,data['native_exploratory']):
        values=[r[k]*100 for k in ['mAP50','mAP50_95','precision','recall','f1']]
        values += [r['latency_ms'],r['fps'],r['parameters']/1e6]
        rows.append(label+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    (OUT/'tables/native_results.tex').write_text(r"""\begin{table*}[!t]
\centering
\caption{Retrospective results on 44 legacy native-resolution exports. Accuracy
entries are percentages; P/R/F1 pool counts at confidence 0.25 and matching IoU
0.50. Latency includes the complete prediction pipeline and WBF. Parameter
counts are stored unfused values. These are selected checkpoints, not seed averages.}
\label{tab:native}
\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrr}
\toprule
Configuration & $\mathrm{mAP}_{50}$ & $\mathrm{mAP}_{50:95}$ & P & R & F1
& ms & FPS & Params (M)\\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    methods=['yolov8s','yolov12s','yolo26s','yolo26s_agwd_sdice','physot_yolo26s','yolo26s_crop_specialist']
    names=['YOLOv8s control','YOLOv12s control','YOLO26s control','AGWD + Squared-Dice',
           'PhysOT-YOLO26s','Crop-trained AGWD + Dice']
    def pm(r,metric):
        return '$'+f"{r[metric+'_mean']*100:.2f}"+r'\pm'+f"{r[metric+'_std']*100:.2f}"+'$'
    rows=[name+' & '+pm(primary[m],'mAP50')+' & '+pm(primary[m],'mAP50_95')+r' \\'
          for m,name in zip(methods,names)]
    (OUT/'tables/paper_dataset_results.tex').write_text(r"""\begin{table}[!t]
\centering
\caption{Available original-label RF100 exports: our rows use the same 41-image
test split and evaluator (mean $\pm$ sample SD, three seeds). Published rows are
contextual; exact protocol equivalence is unverified.}
\label{tab:paper}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{p{.46\columnwidth}rr}
\toprule
Method & $\mathrm{mAP}_{50}$ (\%) & $\mathrm{mAP}_{50:95}$ (\%)\\
\midrule
\multicolumn{3}{l}{\emph{Our shared evaluation protocol}}\\
"""+'\n'.join(rows)+r"""
\midrule
\multicolumn{3}{l}{\emph{Published context only~\cite{zhou2026ensemble}}}\\
YOLOv8s, Table 4 & 54.04 & --\\
Consensus Top-4, Table 9 & 59.63 & --\\
\bottomrule
\end{tabular}
\end{table}
""")
    (OUT/'tables/data_review.tex').write_text(r"""\begin{table}[!t]
\centering
\caption{Recorded r3 edits and quarantined exports. Counts include resized or
transformed views, rather than independent physical components.}
\label{tab:review}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
Split & Class changes & Added & Removed & Quarantined\\
\midrule
Train & 1454 & 83 & 21 & 56\\
Validation & 100 & 12 & 0 & 2\\
Test & 204 & 0 & 4 & 0\\
\midrule
Total & 1758 & 95 & 25 & 58\\
\bottomrule
\end{tabular}
\end{table}
""")
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    classes=['Capacitor','Connector','Electrolytic Capacitor','IC']
    fig,ax=plt.subplots(figsize=(3.5,2.65))
    for offset,key,label,color in [(-.24,'crop','Single','#bd8a30'),
                                   (0,'ensemble2','WBF-2','#42839a'),
                                   (.24,'ensemble4','WBF-4','#173f55')]:
        r=next(r for r in native['results'] if r['id']==key)
        values=[r['accuracy']['per_class'][c]['ap50']*100 for c in classes]
        ax.bar(np.arange(4)+offset,values,width=.23,label=label,color=color,zorder=3)
    ax.set(xticks=range(4),xticklabels=['Capacitor','Connector','Electrolytic\ncapacitor','IC'],
           ylabel='AP50 (%)',ylim=(0,102))
    ax.tick_params(axis='x',labelsize=7)
    ax.grid(axis='y',color='#dde3e8',lw=.5);ax.set_axisbelow(True)
    ax.legend(ncol=3,loc='upper center',fontsize=7,frameon=False,columnspacing=1)
    fig.tight_layout(pad=.5)
    fig.savefig(OUT/'figures/native_class_ap.png',dpi=400,bbox_inches='tight')
    fig.savefig(OUT/'figures/native_class_ap.svg',bbox_inches='tight')
    plt.close(fig)
    for source,destination in [('16_native_ensemble_detections.png','native_detections.png'),
                                ('13_annotation_corrections.png','annotation_review.png')]:
        shutil.copy2(FIG/source,OUT/'figures'/destination)
    sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in [
        'paper/figures/meeting_20261106/figure_manifest.json',
        'results/champion_audit_v1_20261030/summary.json',
        'results/info17_protocol_v1_20261031/summary.json',
        'data_review/native1280_v2_20261106/patches_final_r3.json']}
    (OUT/'provenance.json').write_text(json.dumps(dict(scope='Experimental section only',
        source_hashes=sources,publication_comparison_certified=False,
        independent_generalization_certified=False,r3_accuracy_claimed=False),indent=2)+'\n')
    from revise_paper_figures import main as revise_figures
    revise_figures()
    print(OUT)

if __name__=='__main__':main()
