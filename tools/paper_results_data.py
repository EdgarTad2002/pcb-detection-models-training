"""Audited, CPU-only data preparation for the paper figures notebook.

Inputs are explicit frozen experiments. Pending jobs and diagnostic author-code
AP variants cannot enter the primary comparison through a latest-file search.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ['Capacitor', 'Connector', 'Electrolytic Capacitor', 'IC']
METHODS = ['yolov8s', 'yolov12s', 'yolo26s', 'yolo26s_agwd_sdice',
           'physot_yolo26s', 'yolo26s_crop_specialist']
LABELS = dict(zip(METHODS, ['YOLOv8s', 'YOLOv12s', 'YOLO26s',
    'AGWD + Squared-Dice', 'PhysOT-YOLO26s', 'Crop-trained AGWD + Dice']))
COLORS = dict(zip(METHODS, ['#8b949e', '#577a9b', '#333f4d', '#007f79', '#80628b', '#c38422']))
PRIMARY = 'results/info17_protocol_v1_20261031'
NATIVE = 'results/champion_audit_v1_20261030'
SPEED = 'results/info17_speed_v1_20261031'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pooled(per_class):
    tp, fp, fn = (sum(v[k] for v in per_class.values()) for k in ['tp', 'fp', 'fn'])
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def split_name_overlap(dataset):
    """Conservative filename audit, not a pixel-level duplicate certification."""
    groups=defaultdict(lambda:defaultdict(list))
    for split,items in dataset['splits'].items():
        for r in items['records']:
            view=r['name'].split('.rf.')[0].lower()
            while re.search(r'_(jpg|png|jpeg)$',view):
                view=re.sub(r'_(jpg|png|jpeg)$','',view)
            groups[view][split].append(r['name'])
    return {view:{s:len(names) for s,names in splits.items()}
            for view,splits in groups.items() if len(splits)>1}


def prepare_tables(root):
    root = Path(root)
    manifest = load(root/PRIMARY/'manifest.json')
    entries = [r for r in load(root/PRIMARY/'metrics.json')
               if r['view'] == 'common_ultralytics101' and r['split'] == 'test']
    assert len(entries) == 18
    rows, class_rows = [], []
    for method in METHODS:
        seeds = sorted(r['seed'] for r in entries if r['model'] == method)
        assert seeds == [0, 1, 2], (method, seeds)
        for r in [r for r in entries if r['model'] == method]:
            rows.append(dict(model=method, seed=r['seed'], mAP50=r['mAP50'],
                mAP50_95=r['mAP50_95'], **pooled(r['per_class'])))
            for name, p in r['per_class'].items():
                class_rows.append(dict(model=method, seed=r['seed'], component=name, **p))
    seeds = pd.DataFrame(rows)
    classes = pd.DataFrame(class_rows)
    primary = seeds.groupby('model').agg(
        **{f'{m}_{s}': (m, s) for m in ['mAP50','mAP50_95','precision','recall','f1']
           for s in ['mean','std']}).reindex(METHODS)
    # Independently verify reported means and sample SDs; no best-seed selection.
    reported = load(root/PRIMARY/'summary.json')
    for method in METHODS:
        record = next(r for r in reported if r['model'] == method and
                      r['view'] == 'common_ultralytics101' and r['split'] == 'test')
        assert record['n'] == 3
        for metric in ['mAP50','mAP50_95']:
            for suffix, name in [('mean','mean'),('std','std')]:
                assert np.isclose(primary.loc[method,f'{metric}_{suffix}'],
                                  record[f'{metric}_{name}'],atol=1e-12)
    parameters = load(root/SPEED/'checkpoint_parameter_counts.json')
    speed = load(root/SPEED/'benchmark.json')
    assert not speed['failures']
    checkpoint_map = {r['sha256']:r for r in manifest['checkpoints']}
    speed_rows = []
    for r in speed['records']:
        assert len(r['checkpoints']) == 1
        identity = r['checkpoints'][0]['sha256']
        cp = checkpoint_map[identity]
        samples = r['precision']['fp32']['wall_prediction']['samples_ms']
        assert len(samples) == 100
        latency = float(np.mean(samples))
        params = next(p['parameters'] for p in parameters['paper_unfused']
                      if p['checkpoint_sha256'] == identity)
        speed_rows.append(dict(model=cp['model'],seed=cp['seed'],latency_ms=latency,
                               fps=1000/latency,parameters=params,checkpoint_sha256=identity))
    speed_seeds = pd.DataFrame(speed_rows)
    assert len(speed_seeds) == 18
    speed_summary = speed_seeds.groupby('model').agg(
        latency_ms_mean=('latency_ms','mean'),latency_ms_std=('latency_ms','std'),
        fps_mean=('fps','mean'),fps_std=('fps','std'),parameters=('parameters','first'))
    primary = primary.join(speed_summary)
    native = load(root/NATIVE/'summary.json')
    assert native['status'] == 'completed' and len(native['results']) == 6
    unfused = {p['checkpoint_sha256']:p['unfused_parameters'] for p in parameters['native_unfused']}
    native_rows = []
    for r in native['results']:
        assert r['manifest_id'] == native['manifest']['id']
        a = r['accuracy']
        native_rows.append(dict(id=r['id'],name=r['name'],mAP50=a['mAP50'],mAP50_95=a['mAP50_95'],
            **pooled(a['per_class']),latency_ms=r['speed']['wall_prediction']['mean_ms'],
            fps=r['speed']['fps_from_mean_latency'],
            parameters=sum(unfused[h] for h in r['checkpoint_sha256'])))
    ablation = load(root/'results/clean_main_test_v1/summary.json')
    assert ablation['status'] == 'completed' and ablation['successful_runs'] == 21
    ablation_rows = [dict(model=r['model'],**{k+'_'+s:r['metrics'][k][field]
                        for k in ['mAP50','mAP50_95']
                        for s,field in [('mean','mean'),('std','sample_sd')]})
                     for r in ablation['summary']]
    return dict(primary=primary,seeds=seeds,classes=classes,speed_seeds=speed_seeds,
        native=pd.DataFrame(native_rows),native_raw=native,manifest=manifest,
        ablation=pd.DataFrame(ablation_rows).set_index('model'),ablation_raw=ablation)


def iou_matrix(gt, pred):
    """Float32 IoU, same arithmetic and epsilon as the frozen scorer."""
    gt=np.asarray(gt,dtype=np.float32).reshape(-1,4)
    pred=np.asarray(pred,dtype=np.float32).reshape(-1,4)
    inter=np.maximum(np.minimum(gt[:,None,2:],pred[None,:,2:])-
                     np.maximum(gt[:,None,:2],pred[None,:,:2]),0).prod(2)
    a=(gt[:,2:]-gt[:,:2]).prod(1)
    b=(pred[:,2:]-pred[:,:2]).prod(1)
    return inter/(a[:,None]+b[None,:]-inter+1e-7)


def greedy_pairs(iou, threshold=.5):
    """IoU-ordered one-to-one matching, preserving frozen evaluator ordering."""
    pairs=np.array(np.nonzero(iou>=threshold)).T
    if len(pairs)>1:
        pairs=pairs[iou[pairs[:,0],pairs[:,1]].argsort()[::-1]]
        pairs=pairs[np.unique(pairs[:,1],return_index=True)[1]]
        pairs=pairs[np.unique(pairs[:,0],return_index=True)[1]]
    return pairs


def ap101(recall, precision):
    """Ultralytics 8.4.126 AP, including its duplicate recall endpoint sentinel."""
    mrec=np.r_[0.,recall,recall[-1] if len(recall) else 1.,1.]
    mpre=np.r_[1.,precision,0.,0.]
    envelope=np.maximum.accumulate(mpre[::-1])[::-1]
    x=np.linspace(0,1,101)
    integral=getattr(np,'trapezoid',None) or np.trapz
    return float(integral(np.interp(x,mrec,envelope),x)),mrec,envelope


def resolve_original(root, record, split='test'):
    """Locate only byte-identical old-label inputs, never substitute r3 labels."""
    root=Path(root)
    image=root/'datasets/info17-original-fourclass-v1'/split/'images'/record['name']
    label=root/'datasets/info17-original-fourclass-v1'/split/'labels'/(Path(record['name']).stem+'.txt')
    assert sha(image)==record['image_sha256'],image
    assert sha(label)==record['label_sha256'],label
    return image,label


def boxes_from_label(label, shape=(640,640)):
    values=np.array([list(map(float,l.split())) for l in Path(label).read_text().splitlines()
                     if l.strip()],dtype=np.float32).reshape(-1,5)
    h,w=shape;xywh=values[:,1:]*np.array([w,h,w,h],dtype=np.float32)
    boxes=np.c_[xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2]
    return values[:,0].astype(int),boxes


def load_seed0_predictions(root, tables):
    root=Path(root);manifest=tables['manifest'];data={}
    recorded={r['path']:r['sha256'] for r in load(root/PRIMARY/'raw_prediction_index.json')['raw_files']}
    records=manifest['dataset']['splits']['test']['records']
    gt=[]
    for r in records:
        image,label=resolve_original(root,r)
        cls,boxes=boxes_from_label(label)
        gt.append(dict(record=r,image=image,label=label,cls=cls,boxes=boxes))
    for cp in [r for r in manifest['checkpoints'] if r['seed']==0]:
        rel='predictions/'+cp['job']+'_test_common.json';path=root/PRIMARY/rel
        assert sha(path)==recorded[rel],path
        p=load(path)
        assert p['checkpoint_sha256']==cp['sha256']
        assert p['image_names']==[r['name'] for r in records]
        data[cp['model']]=[np.asarray(a,dtype=np.float32).reshape(-1,6) for a in p['predictions']]
    return gt,data


def curves_and_checks(gt, predictions, expected):
    flags=[];conf=[];classes=[]
    for g,p in zip(gt,predictions,strict=True):
        overlaps=iou_matrix(g['boxes'],p[:,:4])
        overlaps*=g['cls'][:,None]==p[:,5][None,:]
        flag=np.zeros(len(p),dtype=bool)
        pairs=greedy_pairs(overlaps)
        if len(pairs):flag[pairs[:,1]]=True
        flags.append(flag);conf.append(p[:,4]);classes.append(p[:,5])
    flag=np.concatenate(flags);conf=np.concatenate(conf);classes=np.concatenate(classes)
    order=np.argsort(-conf);flag,conf,classes=flag[order],conf[order],classes[order]
    curves=[]
    for c,name in enumerate(CLASSES):
        selected=classes==c;tp=np.cumsum(flag[selected]);fp=np.cumsum(~flag[selected])
        n=sum(int((g['cls']==c).sum()) for g in gt)
        recall=tp/(n+1e-16);precision=tp/(tp+fp+1e-16)
        ap,mrec,mpre=ap101(recall,precision)
        assert np.isclose(ap,expected['per_class'][name]['ap50'],atol=2e-6), (name,ap,expected['per_class'][name]['ap50'])
        # Fixed-confidence P/R require rematching after filtering, as in frozen code.
        total_tp=total_predictions=0
        for g,p in zip(gt,predictions,strict=True):
            p=p[p[:,4]>=.25]
            overlaps=iou_matrix(g['boxes'],p[:,:4])*(g['cls'][:,None]==p[:,5][None,:])
            pairs=greedy_pairs(overlaps)
            total_tp+=sum(p[j,5]==c for _,j in pairs)
            total_predictions+=int((p[:,5]==c).sum())
        assert total_tp==expected['per_class'][name]['tp']
        assert total_predictions-total_tp==expected['per_class'][name]['fp']
        curves.append(dict(component=name,targets=n,ap50=ap,recall=recall,precision=precision,
                           envelope_recall=mrec,envelope_precision=mpre))
    return curves


def confusion(gt,predictions,threshold=.25):
    """Diagnostic class-agnostic spatial matching; rows=prediction, columns=GT.

    This supports off-diagonal classification errors. It is distinct from the
    class-aware matching used for reported AP and operating precision/recall.
    """
    matrix=np.zeros((5,5),dtype=int)
    for g,raw in zip(gt,predictions,strict=True):
        p=raw[raw[:,4]>=threshold]
        pairs=greedy_pairs(iou_matrix(g['boxes'],p[:,:4]))
        matched_gt=set(pairs[:,0]) if len(pairs) else set()
        matched_pred=set(pairs[:,1]) if len(pairs) else set()
        for i,j in pairs:matrix[int(p[j,5]),int(g['cls'][i])]+=1
        for i,c in enumerate(g['cls']):
            if i not in matched_gt:matrix[4,c]+=1
        for j,row in enumerate(p):
            if j not in matched_pred:matrix[int(row[5]),4]+=1
    assert matrix[4,4]==0
    return matrix
