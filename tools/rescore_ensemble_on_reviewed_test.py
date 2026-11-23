#!/usr/bin/env python3
"""Re-score fixed four/five-model predictions against r3's corrected test labels.

The same 44 images, historical checkpoints and inference outputs are retained.
No model training, inference, fusion/weight search or dataset mutation.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.controlled_experiments import digest, file_hash, split_signature
from tools.evaluate_champion_audit import score, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/champion_plus_exp7_reviewed_r3_test_v1_20261110')
    output=parser.parse_args().output.resolve()
    if output.exists():raise FileExistsError(f'Refuse to overwrite: {output}')
    import torch
    torch.set_num_threads(2)
    native=ROOT/'results/champion_audit_v1_20261030'
    addition=ROOT/'results/champion_plus_exp7_v1_20261110'
    plan_path=ROOT/'controlled_experiments/877b4465e678/plan.json'
    plan=json.loads(plan_path.read_text())
    old=json.loads((native/'summary.json').read_text())
    plus=json.loads((addition/'summary.json').read_text())
    old_records=old['manifest']['records']
    corrected=plan['datasets']['clean']['splits']['test']
    settings=old['manifest']['config']['settings']
    if len(old_records)!=44 or len(corrected)!=44:raise ValueError('Wrong test membership')
    if [r['name'] for r in old_records]!=[r['name'] for r in corrected]:raise ValueError('Image names/order differ')
    for before,after in zip(old_records,corrected):
        if before['image_sha256']!=after['image_sha256']:raise ValueError('Image pixels differ')
        for r in [before,after]:
            if file_hash(r['image'])!=r['image_sha256'] or file_hash(r['label'])!=r['label_sha256']:
                raise ValueError('Frozen image or label content changed')
    fingerprint=plan['dataset_fingerprints']['clean']['test']
    if split_signature(corrected)!=fingerprint:raise ValueError('Corrected test fingerprint differs')
    for package,version in old['manifest']['versions'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Wrong scorer environment')
    if settings!=plus['manifest']['settings']:raise ValueError('Four/five model settings differ')
    paths={'four':native/'predictions/ensemble4.json','five':addition/'predictions.json'}
    caches={k:json.loads(p.read_text()) for k,p in paths.items()}
    expected_ids={'four':old['manifest']['id'],'five':plus['manifest']['id']}
    for key,c in caches.items():
        if c['manifest_id']!=expected_ids[key] or c['image_names']!=[r['name'] for r in old_records]:
            raise ValueError('Wrong prediction identity or image order')
    if caches['four']['shapes']!=caches['five']['shapes']:raise ValueError('Coordinate scales differ')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),settings=settings,
        image_count=44,corrected_test_fingerprint=fingerprint,
        legacy_test_fingerprint=old['manifest']['test_fingerprint'],
        changed_label_files=sum(a['label_sha256']!=b['label_sha256'] for a,b in zip(old_records,corrected)),
        same_image_bytes=True,same_predictions=True,
        members=plus['manifest']['members'],weights=plus['manifest']['weights'],
        input_hashes={str(p):file_hash(str(p)) for p in [plan_path,native/'summary.json',addition/'summary.json',*paths.values()]},
        scorer_source_sha256=file_hash(str(ROOT/'tools/evaluate_champion_audit.py')),
        runner_source_sha256=file_hash(str(Path(__file__))),corrected_records=corrected,
        scope='Existing historical four-model ensemble versus adding historical guide Exp7. This does not evaluate the retrained r3 ensemble.',
        disclosure='R3 is a first-pass reviewed candidate, not a fully frozen independent paper benchmark. No further annotation edits or threshold tuning were performed.')
    manifest['id']=digest(manifest);write(output/'manifest.json',manifest)
    legacy={};reviewed={}
    expected={'four':plus['baseline'],'five':plus['five_model']}
    for key,c in caches.items():
        legacy[key]=score(c['predictions'],old_records,c['shapes'],settings)
        for metric in ['mAP50','mAP50_95']:
            if abs(legacy[key][metric]-expected[key][metric])>2e-6:
                raise ValueError('Legacy control fails to reproduce')
        reviewed[key]=score(c['predictions'],corrected,c['shapes'],settings)
    file_hash.cache_clear()
    for r in corrected:
        if file_hash(r['label'])!=r['label_sha256']:raise ValueError('Corrected labels changed during scoring')
    result=dict(status='completed_verified',manifest=manifest,legacy=legacy,corrected_r3=reviewed,
        mAP50_change_percentage_points=100*(reviewed['five']['mAP50']-reviewed['four']['mAP50']),
        mAP50_95_change_percentage_points=100*(reviewed['five']['mAP50_95']-reviewed['four']['mAP50_95']),
        legacy_controls_reproduced=True)
    write(output/'summary.json',result)
    for key in ['four','five']:
        r=reviewed[key]
        print(f"Corrected r3 {key}: AP50={r['mAP50']*100:.4f}%, AP50:95={r['mAP50_95']*100:.4f}%",flush=True)
    print(f"Addition change: {result['mAP50_change_percentage_points']:+.4f} pp AP50",flush=True)


if __name__=='__main__':main()
