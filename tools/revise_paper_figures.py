#!/usr/bin/env python3
"""Reproducible scientific figures for the expanded results section.

Uses saved predictions and labels only. No training, generated PCB imagery,
confidence tuning, or mutation of source results. Prose is never overwritten.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from paper_results_data import (prepare_tables, boxes_from_label, iou_matrix,
    greedy_pairs, sha, CLASSES, load_seed0_predictions, curves_and_checks)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'paper/overleaf_results_section'
EVIDENCE = ROOT/'paper/sources/revision_20261106'
REVIEW = ROOT/'data_review/native1280_v2_20261106'
NAVY, TEAL, GOLD, RED, BLUE = '#193c53', '#008875', '#cc8724', '#d33f55', '#4078ad'
PROVENANCE = {'sources': {}, 'figures': {}, 'limits': {
    'published_protocol_equivalence_verified': False,
    'corrected_dataset_accuracy_claimed': False,
    'qualitative_examples_selected_after_inspecting_predictions': True}}


def read(path):
    path = Path(path)
    PROVENANCE['sources'][str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def save(fig, name, details):
    fig.savefig(OUT/'figures'/f'{name}.png', dpi=400, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT/'figures'/f'{name}.svg', bbox_inches='tight', facecolor='white')
    PROVENANCE['figures'][name] = details
    plt.close(fig)


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dde5eb', lw=.6, zorder=0)
    ax.set_axisbelow(True)


def inside(boxes, roi):
    centers = (boxes[:, :2] + boxes[:, 2:])/2
    return ((centers[:, 0] >= roi[0]) & (centers[:, 0] < roi[2]) &
            (centers[:, 1] >= roi[1]) & (centers[:, 1] < roi[3]))


def native_inputs(tables):
    summary = tables['native_raw']
    records = summary['manifest']['records']
    caches = {k: read(ROOT/'results/champion_audit_v1_20261030/predictions'/f'{k}.json')
              for k in ['crop', 'ensemble2', 'ensemble4']}
    result = []
    for i, r in enumerate(records):
        image = REVIEW/'source/test/images'/r['name']
        label = image.parent.parent/'labels'/(image.stem+'.txt')
        assert sha(image) == r['image_sha256'] and sha(label) == r['label_sha256']
        c, b = boxes_from_label(label, caches['crop']['shapes'][i])
        ps = {}
        for key, cache in caches.items():
            assert cache['image_names'][i] == r['name']
            assert cache['manifest_id'] == summary['manifest']['id']
            p = np.asarray(cache['predictions'][i], np.float32).reshape(-1, 6)
            ps[key] = p[p[:, 4] >= .25]
        result.append(dict(record=r, image=image, cls=c, boxes=b, predictions=ps))
    return result


def local_match(g, roi, key):
    gi = np.flatnonzero(inside(g['boxes'], roi))
    p = g['predictions'][key]
    p = p[inside(p[:, :4], roi)]
    pairs = greedy_pairs(iou_matrix(g['boxes'][gi], p[:, :4]) *
                         (g['cls'][gi, None] == p[None, :, 5]))
    mg = set(pairs[:, 0]) if len(pairs) else set()
    mp = set(pairs[:, 1]) if len(pairs) else set()
    return gi, p, mg, mp


def choose_examples(gs):
    candidates = []
    for i, g in enumerate(gs):
        im = Image.open(g['image']); W, H = im.size
        for size in [220, 320, 450]:
            if min(W, H) < size:
                continue
            for y in np.linspace(0, H-size, 8).astype(int):
                for x in np.linspace(0, W-size, 8).astype(int):
                    roi = [int(x), int(y), int(x+size), int(y+size)]
                    n = int(inside(g['boxes'], roi).sum())
                    if not 4 <= n <= 18:
                        continue
                    counts = {k: len(local_match(g, roi, k)[2]) for k in g['predictions']}
                    gain = counts['ensemble4'] - counts['crop']
                    candidates.append(dict(image_index=i, roi=roi, targets=n,
                                           matched=counts, gain=gain))
    # Explicit outcome-selected illustrations; these do not estimate average gains.
    recovery = max(candidates, key=lambda r: (r['gain'], -r['targets'], -r['image_index']))
    failure = min((r for r in candidates if r['targets'] <= 10),
                  key=lambda r: (r['gain'], r['targets'], r['image_index']))
    return recovery, failure


def rect(ax, b, color, style='-', lw=1.6):
    ax.add_patch(Rectangle(b[:2], b[2]-b[0], b[3]-b[1], fill=False,
                           edgecolor=color, linewidth=lw, linestyle=style))


def detection_figure(tables):
    gs = native_inputs(tables)
    cases = choose_examples(gs)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.75))
    details = []
    for row, case in enumerate(cases):
        g = gs[case['image_index']]; roi = case['roi']
        im = Image.open(g['image']).convert('RGB')
        gi = np.flatnonzero(inside(g['boxes'], roi))
        case_details = dict(case, name=g['record']['name'], counts={})
        for col, key in enumerate([None, 'crop', 'ensemble2', 'ensemble4']):
            ax = axes[row, col]; ax.imshow(im)
            if key is None:
                for b in g['boxes'][gi]: rect(ax, b, BLUE, lw=1.3)
                title = f'Reference: {len(gi)} labels'
            else:
                gi, p, mg, mp = local_match(g, roi, key)
                for j, b in enumerate(g['boxes'][gi]):
                    if j not in mg: rect(ax, b, RED, '--', 1.4)
                for j, b in enumerate(p[:, :4]):
                    rect(ax, b, TEAL if j in mp else GOLD, '-', 1.7)
                counts = dict(tp=len(mg), fn=len(gi)-len(mg), fp=len(p)-len(mp))
                case_details['counts'][key] = counts
                title = f"Matched {counts['tp']} | Missed {counts['fn']}\nUnmatched {counts['fp']}"
            ax.set_xlim(roi[0], roi[2]); ax.set_ylim(roi[3], roi[1]); ax.axis('off')
            ax.set_title(title, fontsize=6.8, pad=4, color=NAVY)
            if row == 0:
                ax.text(.5, 1.19, ['Existing labels','Crop-trained single','Two-model WBF','Four-model WBF'][col],
                        transform=ax.transAxes, ha='center', fontsize=8, fontweight='bold')
        axes[row, 0].text(-.07, .5, ['(a) Recovery example', '(b) Failure example'][row],
            transform=axes[row, 0].transAxes, rotation=90, ha='right', va='center', fontsize=8)
        details.append(case_details)
    handles = [Line2D([0],[0],color=BLUE,lw=2,label='Existing reference box'),
        Line2D([0],[0],color=TEAL,lw=2,label='Matched prediction'),
        Line2D([0],[0],color=RED,lw=2,ls='--',label='Missed reference'),
        Line2D([0],[0],color=GOLD,lw=2,label='Unmatched prediction')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7,
               columnspacing=1, handlelength=1.6, bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(left=.05,right=.995,top=.9,bottom=.08,wspace=.055,hspace=.24)
    save(fig, 'native_detections', dict(cases=details, confidence=.25, matching_iou=.5,
        selection='Maximum and minimum local matched-reference gain of WBF-4 versus crop, over a deterministic grid; outcome selected.',
        scoring='Local class-aware one-to-one matching; boxes enter by center inside the displayed ROI. Existing labels may contain errors.'))


def paper_comparison(t):
    names = ['YOLOv8s','YOLOv12s','YOLO26s','AGWD + Dice','PhysOT','Crop + AGWD/Dice']
    methods = ['yolov8s','yolov12s','yolo26s','yolo26s_agwd_sdice','physot_yolo26s','yolo26s_crop_specialist']
    p=t['primary'];fig,axs=plt.subplots(1,2,figsize=(7.15,2.8),gridspec_kw={'width_ratios':[1.65,1]})
    vals=[p.loc[m,'mAP50_mean']*100 for m in methods]
    err=[p.loc[m,'mAP50_std']*100 for m in methods]
    axs[0].barh(range(6),vals,xerr=err,color=[NAVY]*5+[TEAL],capsize=2,zorder=3)
    axs[0].set(yticks=range(6),yticklabels=names,xlim=(0,65),xlabel='mAP50 (%)')
    axs[0].invert_yaxis();axs[0].set_title('(a) Our common test protocol',fontsize=9)
    for i,v in enumerate(vals):axs[0].text(v+err[i]+.8,i,f'{v:.2f}',va='center',fontsize=7)
    axs[1].barh([0,1],[54.04,59.63],color=['#8997a1',GOLD],hatch='//',zorder=3)
    axs[1].set(yticks=[0,1],yticklabels=['Best single\nYOLOv8s','Best ensemble\nConsensus Top-4'],xlim=(0,69),ylim=(1.8,-.8),xlabel='Reported mAP50 (%)')
    axs[1].tick_params(axis='y',labelsize=7)
    axs[1].set_title('(b) Published context',fontsize=9)
    for i,v in enumerate([54.04,59.63]):axs[1].text(v+.6,i,f'{v:.2f}',va='center',fontsize=7)
    for ax in axs:clean(ax);ax.grid(axis='x',color='#dde5eb',lw=.6);ax.grid(axis='y',visible=False)
    fig.tight_layout(w_pad=2)
    save(fig,'paper_comparison',dict(our_protocol='41 test exports; three-seed mean and sample SD',
        published={'best_single':54.04,'best_ensemble':59.63,'source':'Zhou and Agaian, Tables 4 and 9'},
        direct_superiority_claim=False))
    rows=[]
    for m,n in zip(methods,names):
        r=p.loc[m]
        values=[f"${r[k+'_mean']*100:.2f}\\pm{r[k+'_std']*100:.2f}$" for k in ['mAP50','mAP50_95','precision','recall','f1']]
        rows.append(n+' & '+' & '.join(values)+r' \\')
    (OUT/'tables/paper_dataset_results.tex').write_text(r'''\begin{table*}[!t]
\centering
\caption{Original-label RF100 exports at 640 pixels. Our methods share the same 41-image test split and evaluator (mean $\pm$ sample SD, seeds 0/1/2). P/R/F1 use pooled counts at confidence 0.25 and matching IoU 0.50. Published rows use the authors' protocol and are contextual; a direct improvement claim is unverified.}
\label{tab:paper}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}\toprule
Method & $\mathrm{mAP}_{50}$ & $\mathrm{mAP}_{50:95}$ & P & R & F1\\\midrule
\multicolumn{6}{l}{\emph{Our common evaluation protocol (all entries in percent)}}\\
'''+ '\n'.join(rows)+r'''
\midrule
\multicolumn{6}{l}{\emph{Published context only~\cite{zhou2026ensemble}; P/R thresholds may differ}}\\
YOLOv8s, best single (Table 4) & 54.04 & -- & 60.6 & 51.8 & --\\
Consensus Top-4, best ensemble (Table 9) & 59.63 & -- & 63.2 & 59.3 & --\\
\bottomrule\end{tabular}\end{table*}
''')
    # Frozen seed-0 PR curves; verify them against the original saved AP results.
    gt,preds=load_seed0_predictions(ROOT,t)
    expected=read(ROOT/'results/info17_protocol_v1_20261031/metrics.json')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.55))
    for method,label,color in [('yolov8s','YOLOv8s',BLUE),('yolo26s','YOLO26s',NAVY),
                               ('yolo26s_crop_specialist','Crop + AGWD/Dice',TEAL)]:
        e=next(r for r in expected if r['model']==method and r['seed']==0 and r['split']=='test' and r['view']=='common_ultralytics101')
        curves=curves_and_checks(gt,preds[method],e)
        for ax,c in zip(axs,[0,3]):
            r=curves[c]
            ax.plot(r['envelope_recall'],r['envelope_precision'],lw=1.3,color=color,label=label)
    for ax,title in zip(axs,['(a) Capacitor','(b) Integrated circuit']):
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision',title=title);clean(ax)
    axs[1].legend(fontsize=7,frameon=False,loc='upper right')
    fig.tight_layout()
    save(fig,'paper_pr_curves',dict(seed=0,classes=['Capacitor','IC'],AP50_independently_checked=True,
        scope='Illustrative seed only; mean results use all three seeds.'))


def sweeps(t):
    metadata=read(EVIDENCE/'sweep_launch_metadata.json')
    settings={}
    for lines in metadata.values():
        text='\n'.join(lines)
        if 'Multi-Stage Squared-Dice' not in text:continue
        name=re.search('Exploration: (\\S+)',text).group(1)
        lam=float(re.search(r'lambda_dice\s*=\s*([\d.]+)',text).group(1))
        smooth=float(re.search(r'smooth_dice\s*=\s*([\d.]+)',text).group(1))
        weights=re.search(r'stage_weights =([^\n]*)',text).group(1).strip()
        settings[name]=dict(lam=lam,smooth=smooth,stage_weights=[float(x) for x in weights.split()] if weights else None)
    rows=[]
    for name,params in settings.items():
        r=read(ROOT/'results'/f'{name}.json')
        assert r['seed']==0 and r['eval_split']=='test' and r['epochs']==100 and r['imgsz']==1280 and r['batch']==8
        assert r['data']=='datasets/pcb-native-res-cleaned-1280/data.yaml'
        assert r['lambda_dice']==params['lam']
        rows.append(dict(name=name,**params,ap50=r['mAP50']*100,ap95=r['mAP50_95']*100,
                         capacitor=r['per_class_ap50']['Capacitor']*100))
    rows.sort(key=lambda r:(r['stage_weights'] is not None,r['smooth'],r['lam']))
    assert len(rows)==7
    pd.DataFrame(rows).to_csv(OUT/'tables/hyperparameter_data.csv',index=False)
    latex=[]
    for r in rows:
        stage='Uniform' if r['stage_weights'] is None else '0.5/0.3/0.2'
        latex.append(f"{r['lam']:g} & {r['smooth']:g} & {stage} & {r['ap50']:.2f} & {r['ap95']:.2f} & {r['capacitor']:.2f}"+r' \\')
    (OUT/'tables/hyperparameters.tex').write_text(r'''\begin{table}[!t]
\centering\caption{Historical AGWD--Squared-Dice configurations on legacy test labels. Seed 0 only; percentages. These are development observations, not independent parameter-selection evidence.}
\label{tab:sweep}\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{rrlrrr}\toprule
$\lambda$ & $\epsilon$ & Stage weights & AP50 & AP50:95 & Cap AP50\\\midrule
'''+ '\n'.join(latex)+r'''
\bottomrule\end{tabular}\end{table}
''')
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.9))
    uniform=[r for r in rows if r['stage_weights'] is None and r['smooth']==.01]
    for key,label,color in [('ap50','mAP50',TEAL),('ap95','mAP50:95',BLUE),('capacitor','Capacitor AP50',GOLD)]:
        axs[0,0].plot([r['lam'] for r in uniform],[r[key] for r in uniform],'o-',color=color,label=label,ms=4)
    axs[0,0].set(xlabel=r'$\lambda$ (uniform weights, $\epsilon=0.01$)',ylabel='AP (%)',title='(a) Loss coefficient');axs[0,0].legend(fontsize=6.5,frameon=False)
    for stage,label,color in [(False,'Uniform stages',TEAL),(True,'Weights 0.5/0.3/0.2',NAVY)]:
        selected=[r for r in rows if (r['stage_weights'] is not None)==stage and r['lam']==.5]
        axs[0,1].plot([r['smooth'] for r in selected],[r['ap50'] for r in selected],'o-',color=color,label=label,ms=4)
    axs[0,1].set(xscale='log',xticks=[.01,.1,1],xticklabels=['0.01','0.1','1.0'],xlabel=r'$\epsilon$ (fixed $\lambda=0.5$)',ylabel='mAP50 (%)',title='(b) Smoothing within each stage policy')
    axs[0,1].legend(fontsize=6.5,frameon=False)
    paired=[next(r for r in rows if r['lam']==.5 and r['smooth']==.01 and (r['stage_weights'] is not None)==b) for b in [False,True]]
    for j,(key,label,color) in enumerate([('ap50','mAP50',TEAL),('ap95','mAP50:95',BLUE),('capacitor','Cap AP50',GOLD)]):
        axs[1,0].bar(np.arange(2)+(j-1)*.23,[r[key] for r in paired],width=.22,label=label,color=color)
    axs[1,0].set(xticks=[0,1],xticklabels=['Uniform','0.5/0.3/0.2'],ylabel='AP (%)',title=r'(c) Stage weighting ($\lambda=0.5,\epsilon=0.01$)');axs[1,0].legend(fontsize=6,frameon=False,ncol=3,loc='lower center')
    for r,color in zip(uniform,[BLUE,GOLD,TEAL]):
        path=ROOT/'runs'/r['name']/'pcb-filtered/results.csv'
        PROVENANCE['sources'][str(path.relative_to(ROOT))]=sha(path)
        history=pd.read_csv(path);history.columns=history.columns.str.strip()
        assert len(history)==100
        axs[1,1].plot(history['epoch'],history['metrics/mAP50(B)']*100,lw=.9,color=color,label=f"lambda={r['lam']:g}")
    axs[1,1].set(xlabel='Training epoch',ylabel='Validation mAP50 (%)',title='(d) Saved validation histories');axs[1,1].legend(fontsize=6.5,frameon=False)
    for ax in axs.flat:clean(ax);ax.title.set_fontsize(8)
    fig.tight_layout(h_pad=1.8,w_pad=1.8)
    save(fig,'hyperparameter_sensitivity',dict(configurations=rows,seed=0,limits='Historical test-guided development; no interpolation to untested values; lambda=1 is staged, excluded from uniform-lambda curve.'))
    # Repeated-seed controls use a separate no-extra-online-augmentation recipe.
    raw=t['ablation_raw'];sel=['yolo26s','yolo26s_sdice','yolo26s_agwd_sdice_l035','yolo26s_agwd_sdice']
    names=['YOLO26s control','Squared-Dice, 0.50','AGWD + Dice, 0.35','AGWD + Dice, 0.50']
    text=[];evidence=[]
    for model,name in zip(sel,names):
        metrics=next(r['metrics'] for r in raw['summary'] if r['model']==model)
        vals=[]
        for key in ['mAP50','mAP50_95','per_class_ap50/Capacitor']:
            q=metrics[key];vals.append(f"${q['mean']:.2f}\\pm{q['sample_sd']:.2f}$")
        text.append(name+' & '+' & '.join(vals)+r' \\');evidence.append(dict(model=model,metrics=metrics))
    (OUT/'tables/controlled_loss.tex').write_text(r'''\begin{table}[!t]\centering
\caption{Separate repeated-seed native study: mean $\pm$ sample SD, three seeds, no extra online augmentation. Rectangular batch-8 evaluation differs from Table~\ref{tab:native}. Accuracy entries are percentages.}
\label{tab:controlled}\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrr}\toprule
Recipe ($\lambda$) & AP50 & AP50:95 & Cap AP50\\\midrule
'''+ '\n'.join(text)+r'''
\bottomrule\end{tabular}\end{table}
''')
    PROVENANCE['figures']['controlled_loss_table']=evidence


def image_record_image(record):
    split,name=record['id'].split('/',1)
    image=REVIEW/'source'/split/'images'/name
    assert sha(image)==record['image_sha256']
    return Image.open(image).convert('RGB')


def xyxy(s, W, H):
    a=list(map(float,s.split()));_,x,y,w,h=a
    return np.array([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H])


def crop_for_box(im,b,pad=.6):
    cx,cy=(b[:2]+b[2:])/2;size=max(b[2:]-b[:2])*(1+2*pad)
    return [max(0,cx-size/2),max(0,cy-size/2),min(im.width,cx+size/2),min(im.height,cy+size/2)]


def taxonomy_figure():
    index=read(EVIDENCE/'evidence_index.json')
    r=index['selected'][0]
    p1=EVIDENCE/r['image'];l1=EVIDENCE/r['label']
    p2=next(EVIDENCE.glob('datasets/pcb-filtered-yolov8/*/images/Arty_Top*'))
    l2=p2.parent.parent/'labels'/(p2.stem+'.txt')
    examples=[(p1,l1,1,'Capacitor Jumper (1)'),(p2,l2,41,'Capacitor (2)')]
    fig,axs=plt.subplots(1,4,figsize=(7.1,2.0));details=[]
    for row,(image,label,line,name) in enumerate(examples):
        im=Image.open(image).convert('RGB');s=label.read_text().splitlines()[line-1]
        assert int(s.split()[0])==[1,2][row]
        b=xyxy(s,im.width,im.height);roi=crop_for_box(im,b,pad=.65)
        for j in range(2):
            ax=axs[row*2+j];ax.imshow(im);rect(ax,b,GOLD if j==0 else TEAL,lw=2)
            ax.set_xlim(roi[0],roi[2]);ax.set_ylim(roi[3],roi[1]);ax.axis('off')
            ax.set_title(name if j==0 else 'Unified: Capacitor (0)',fontsize=8)
        details.append(dict(image=str(image.relative_to(ROOT)),label=str(label.relative_to(ROOT)),line=line,raw=s,canonical_class=0))
        for path in [image,label]:PROVENANCE['sources'][str(path.relative_to(ROOT))]=sha(path)
    fig.subplots_adjust(wspace=.08,top=.88,bottom=.03)
    save(fig,'taxonomy_consolidation',dict(examples=details,raw_class_counts=index['raw_class_counts'],
        note='Examples illustrate canonical label mapping, not an exhaustive claim that both source categories are error-free. Class 22 iC has no instances in the scanned source export.'))


def correction_figure():
    inventory=read(REVIEW/'audit/inventory.json');records={r['id']:r for r in inventory['records']}
    patches=read(REVIEW/'patches_final_r3.json')
    replacements=[p for p in patches if p['action']=='replace' and p['id'].startswith('test/')]
    def area(p):
        b=next(b for b in records[p['id']]['boxes'] if b['line']==p['line'])
        return b['xywh'][2]*b['xywh'][3]
    replacement=max(replacements,key=area)
    removed=next(p for p in patches if p['action']=='delete')
    missing=next(r for r in inventory['records'] if r['id']=='train/pcb149rec1_jpg.rf.326922824f63217a6e1021689887440a.jpg')
    fig,axs=plt.subplots(3,2,figsize=(5.6,7.0));details=[]
    for row,item in enumerate([replacement,missing,removed]):
        record=item if row==1 else records[item['id']]
        im=image_record_image(record);W,H=im.size
        if row==0:
            b=xyxy(item['expected'],W,H);roi=crop_for_box(im,b,.35);before=[b];after=[b]
            titles=['Original: general capacitor','Reviewed: electrolytic capacitor']
        elif row==1:
            add=[p for p in patches if p['id']==record['id'] and p['action']=='add']
            after=[xyxy(p['replacement'],W,H) for p in add];before=[];roi=[735,0,1150,460]
            titles=['Original: missing component boxes','Reviewed: boxes added']
        else:
            b=xyxy(item['expected'],W,H);roi=crop_for_box(im,b,1.3);before=[b];after=[]
            titles=['Original: box on empty footprint','Reviewed: incorrect box removed']
        for col,boxes in enumerate([before,after]):
            ax=axs[row,col];ax.imshow(im)
            for b in boxes:rect(ax,b,GOLD if col==0 else TEAL,lw=2)
            ax.set_xlim(roi[0],roi[2]);ax.set_ylim(roi[3],roi[1]);ax.axis('off');ax.set_title(titles[col],fontsize=8)
        axs[row,0].text(-.06,.5,['(a) Class correction','(b) Missing annotations','(c) Empty-footprint label'][row],
            transform=axs[row,0].transAxes,rotation=90,ha='right',va='center',fontsize=8)
        details.append(dict(id=record['id'],roi=roi,action=['replace','add','delete'][row]))
    fig.subplots_adjust(left=.07,right=.99,top=.97,bottom=.02,hspace=.18,wspace=.07)
    save(fig,'annotation_review',dict(examples=details,scope='Recorded annotation edits, not predictions or measured accuracy improvements.'))


def main():
    (OUT/'figures').mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    t=prepare_tables(ROOT)
    for p in ['results/champion_audit_v1_20261030/summary.json','results/clean_main_test_v1/summary.json',
              'results/info17_protocol_v1_20261031/metrics.json','tools/rectify_dataset.py']:
        PROVENANCE['sources'][p]=sha(ROOT/p)
    templates=sorted((ROOT/'paper_images').glob('*.png'))
    PROVENANCE['visual_reference_templates']={str(p.relative_to(ROOT)):sha(p) for p in templates}
    PROVENANCE['template_use']='Layout inspiration only: enlarged comparisons, parameter/metric plots, configuration tables, and paired annotation examples. No published result images are reused as our outputs.'
    class_rows=[]
    names=['YOLOv12s','Crop-trained YOLO26s','AGWD + Squared-Dice','PhysOT-YOLO26s',
           'WBF, two models','WBF, four models']
    for name,r in zip(names,t['native_raw']['results']):
        class_rows.append(name+' & '+' & '.join(f"{r['accuracy']['per_class'][c]['ap50']*100:.2f}" for c in CLASSES)+r' \\')
    (OUT/'tables/native_classes.tex').write_text(r'''\begin{table}[!t]\centering
\caption{Per-class AP50 (\%) in the native checkpoint audit. Only 18 electrolytic-capacitor boxes occur in these legacy test labels.}
\label{tab:nativeclasses}\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}\toprule
Configuration & Cap. & Conn. & Elec. & IC\\\midrule
'''+ '\n'.join(class_rows)+r'''
\bottomrule\end{tabular}\end{table}
''')
    detection_figure(t)
    paper_comparison(t)
    sweeps(t)
    taxonomy_figure()
    correction_figure()
    (OUT/'revision_provenance.json').write_text(json.dumps(PROVENANCE,indent=2)+'\n')
    print('Generated revised figures, tables and provenance:',OUT)


if __name__=='__main__':main()
