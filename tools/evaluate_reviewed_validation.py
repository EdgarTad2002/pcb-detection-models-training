#!/usr/bin/env python3
"""Compare frozen old and retrained checkpoints on the unchanged r3 validation set.

Validation only. No training, test inference, confidence sweeps or ensembles.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.controlled_experiments import NAMES, file_hash, split_signature

PLAN = ROOT/'controlled_experiments/877b4465e678/plan.json'
DEFAULT_OUTPUT = ROOT/'results/reviewed_r3_old_vs_retrained_v2_20261110'
SETTINGS = dict(split='val', imgsz=1280, batch=8, workers=8, rect=True,
                conf=.001, iou=.5, max_det=1000, half=False, augment=False,
                agnostic_nms=False, classes=None, quantize=None)


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def verify_records(records, expected):
    file_hash.cache_clear()
    for r in records:
        if file_hash(str(Path(r['image']).resolve())) != r['image_sha256']:
            raise ValueError(f"Validation image changed: {r['name']}")
        if file_hash(str(Path(r['label']).resolve())) != r['label_sha256']:
            raise ValueError(f"Validation labels changed: {r['name']}")
    if split_signature(records) != expected:
        raise ValueError('Wrong validation membership or fingerprint')


def prepare(output):
    import yaml
    if output.exists():
        raise FileExistsError(f'Already prepared: {output}; use run to resume')
    plan=json.loads(PLAN.read_text())
    required=plan['config']['ultralytics_version']
    if importlib.metadata.version('ultralytics') != required:
        raise ValueError(f'Use ultralytics=={required}')
    records=plan['datasets']['clean']['splits']['val']
    fingerprint=plan['dataset_fingerprints']['clean']['val']
    if len(records)!=78:
        raise ValueError('Expected 78 corrected validation exports')
    verify_records(records,fingerprint)
    source_yaml=Path(plan['output'])/'datasets/clean/data.yaml'
    source_cfg=yaml.safe_load(source_yaml.read_text())
    if source_cfg['names']!=NAMES:
        raise ValueError('Wrong canonical classes')
    source_list=Path(source_cfg['val'])
    if source_list.read_text().splitlines()!=[r['image'] for r in records]:
        raise ValueError('Frozen pilot YAML membership differs from plan')
    cfg=json.loads((ROOT/'configs/champion_audit.json').read_text())
    mapping={'yolo12':'yolov12s','crop':'yolo26s_crop_specialist',
             'agwd_sdice':'yolo26s_agwd_sdice','physot':'physot_yolo26s'}
    cases=[]
    for r in cfg['models']:
        checkpoint=(ROOT/r['path']).resolve()
        if file_hash(str(checkpoint))!=r['sha256']:
            raise ValueError(f"Old champion differs from audited checkpoint: {r['id']}")
        cases.append(dict(id='old_'+r['id'],arm='old',model=mapping[r['id']],
                          checkpoint=str(checkpoint),checkpoint_sha256=r['sha256']))
    # The historical result JSON is archived on the cluster; freeze the retained
    # original checkpoint directly rather than restoring or substituting weights.
    checkpoint=ROOT/'runs/yolo26s_native_cleaned_baseline_cappaste_1280/pcb-filtered/weights/best.pt'
    cases.append(dict(id='old_yolo26s',arm='old',model='yolo26s',checkpoint=str(checkpoint),
                      checkpoint_sha256=file_hash(str(checkpoint))))
    for job in plan['jobs']:
        record=Path(plan['output'])/'validation'/(job['key']+'.json')
        previous=json.loads(record.read_text())
        checkpoint=ROOT/'runs'/job['key']/'pcb-filtered/weights/best.pt'
        if previous['weights']!=str(checkpoint) or previous['eval_split']!='val' or previous['seed']!=0:
            raise ValueError('Retrained checkpoint differs from completed pilot report')
        cases.append(dict(id='retrained_'+job['model'],arm='retrained',model=job['model'],
                          checkpoint=str(checkpoint),checkpoint_sha256=file_hash(str(checkpoint)),
                          pilot_record=str(record),pilot_record_sha256=file_hash(str(record)),
                          pilot_mAP50=previous['mAP50'],pilot_mAP50_95=previous['mAP50_95']))
    if len(cases)!=10 or len({(r['model'],r['arm']) for r in cases})!=10:
        raise ValueError('Require five distinct old/retrained pairs')
    paths=[Path(__file__),ROOT/'tools/controlled_experiments.py',ROOT/'tools/benchmark_speed.py',*ROOT.glob('*.py')]
    manifest=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        plan=str(PLAN),plan_sha256=file_hash(str(PLAN)),records=records,
        validation_fingerprint=fingerprint,validation_image_count=len(records),
        source_yaml=str(source_yaml),source_yaml_sha256=file_hash(str(source_yaml)),
        source_list=str(source_list),source_list_sha256=file_hash(str(source_list)),
        settings=SETTINGS,names=NAMES,cases=cases,
        source_hashes={str(p):file_hash(str(p)) for p in paths},
        versions={p:importlib.metadata.version(p) for p in ['ultralytics','torch','torchvision','numpy']},
        disclosure='Matched evaluation, not isolated label-correction causality. Old training recipes and data differ, and old models may have seen related held-out photographs. Retrained checkpoints were selected on this validation set. Seed-0 pilot; no test inference.')
    write(output/'manifest.json',manifest)
    print(f'Prepared {len(cases)} exact checkpoints on {len(records)} validation exports: {output}',flush=True)


def verify(manifest):
    verify_records(manifest['records'],manifest['validation_fingerprint'])
    for field in ['plan','source_yaml','source_list']:
        if file_hash(manifest[field])!=manifest[field+'_sha256']:
            raise ValueError(f'Frozen {field} changed')
    for path,expected in manifest['source_hashes'].items():
        if file_hash(path)!=expected:raise ValueError(f'Frozen evaluation source changed: {path}')
    for package,expected in manifest['versions'].items():
        if importlib.metadata.version(package)!=expected:raise ValueError('Environment changed')
    for c in manifest['cases']:
        if file_hash(c['checkpoint'])!=c['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        for prefix in ['source_record','pilot_record']:
            if prefix in c and file_hash(c[prefix])!=c[prefix+'_sha256']:
                raise ValueError('Source result changed')


def summarize(manifest, results):
    if len(results)!=len(manifest['cases']):raise ValueError('Incomplete comparison')
    indexed={}
    for r in results:
        expected=next(c for c in manifest['cases'] if c['id']==r['id'])
        if r['settings']!=manifest['settings'] or r['validation_fingerprint']!=manifest['validation_fingerprint'] or r['checkpoint_sha256']!=expected['checkpoint_sha256']:
            raise ValueError('Mixed checkpoints, evaluation settings or annotations')
        key=(r['model'],r['arm'])
        if key in indexed:raise ValueError('Duplicate comparison arm')
        indexed[key]=r
    pairs=[]
    for model in sorted({c['model'] for c in manifest['cases']}):
        old=indexed[(model,'old')];new=indexed[(model,'retrained')]
        pairs.append(dict(model=model,old_mAP50_percent=old['mAP50']*100,
            retrained_mAP50_percent=new['mAP50']*100,
            mAP50_change_percentage_points=(new['mAP50']-old['mAP50'])*100,
            old_mAP50_95_percent=old['mAP50_95']*100,
            retrained_mAP50_95_percent=new['mAP50_95']*100,
            mAP50_95_change_percentage_points=(new['mAP50_95']-old['mAP50_95'])*100))
    return dict(status='completed',settings=manifest['settings'],
                validation_fingerprint=manifest['validation_fingerprint'],
                validation_image_count=manifest['validation_image_count'],
                pairs=pairs,results=results,disclosure=manifest['disclosure'])


def runtime_observations(validator):
    # This validator keeps AutoBackend local to __call__, not on self.model.
    # args.quantize is updated from the actual backend precision before callbacks.
    return dict(images_seen=int(validator.seen),fp16=validator.args.quantize==16,
                rect=bool(validator.args.rect),split=validator.args.split,
                target_counts=[int(v) for v in validator.metrics.nt_per_class])


def run(output, device):
    import torch
    from ultralytics import YOLO
    from tools.benchmark_speed import register_modules
    manifest=json.loads((output/'manifest.json').read_text());verify(manifest)
    register_modules()
    results=[]
    for case in manifest['cases']:
        path=output/'metrics'/(case['id']+'.json')
        if path.exists():
            results.append(json.loads(path.read_text()));continue
        print('Evaluating '+case['id'],flush=True)
        model=YOLO(case['checkpoint'])
        if [model.names[i] for i in range(len(model.names))]!=NAMES:
            raise ValueError('Checkpoint class mapping is not canonical')
        observed={}
        def on_end(validator):
            observed.update(runtime_observations(validator))
        model.add_callback('on_val_end',on_end)
        metrics=model.val(data=manifest['source_yaml'],device=device,**manifest['settings'],
                          project=str(output/'artifacts'),name=case['id'],exist_ok=False,
                          plots=False,save_json=True,verbose=False)
        if observed['images_seen']!=78 or observed['fp16'] or not observed['rect'] or observed['split']!='val':
            raise ValueError(f'Runtime evaluation deviates from matched settings: {observed}')
        counts=[0]*4
        for r in manifest['records']:
            for line in Path(r['label']).read_text().splitlines():
                if line.strip():counts[int(line.split()[0])]+=1
        if observed['target_counts']!=counts:raise ValueError('Validator targets differ from frozen labels')
        indices=list(metrics.box.ap_class_index)
        per_class={}
        for i,name in enumerate(NAMES):
            ap=metrics.box.all_ap[indices.index(i)] if i in indices else [0.]*10
            per_class[name]=dict(ap50=float(ap[0]),ap50_95=float(sum(ap)/len(ap)),targets=counts[i])
        result=dict(**case,settings=manifest['settings'],validation_fingerprint=manifest['validation_fingerprint'],
                    mAP50=float(metrics.box.map50),mAP50_95=float(metrics.box.map),
                    per_class=per_class,runtime=observed)
        if case['arm']=='retrained':
            result['pilot_reproduction_deltas']={k:result[k]-case['pilot_'+k] for k in ['mAP50','mAP50_95']}
            if any(abs(v)>5e-4 for v in result['pilot_reproduction_deltas'].values()):
                raise ValueError('Retrained score differs from pilot by more than 0.05 percentage points')
        write(path,result);results.append(result)
        del model,metrics;gc.collect();torch.cuda.empty_cache()
    verify(manifest)
    summary=summarize(manifest,results);summary['manifest']=manifest
    write(output/'summary.json',summary)
    for row in summary['pairs']:
        print(f"{row['model']}: old={row['old_mAP50_percent']:.2f}%, retrained={row['retrained_mAP50_percent']:.2f}%, change={row['mAP50_change_percentage_points']:+.2f} pp",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--device',default='0');args=p.parse_args()
    if args.action=='prepare':prepare(args.output.resolve())
    else:run(args.output.resolve(),args.device)


if __name__=='__main__':main()
