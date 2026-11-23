#!/usr/bin/env python3
"""Evaluate the guide models on the exact legacy 1280 test benchmark.

Includes historical guide weights and all six new original-label 640-trained
weights. Fixed two-member WBFs use the guide's 1.30/1.25 weights and the common
audit's merge IoU .50. No tuning or training; original audit is never modified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.controlled_experiments import digest, file_hash
from tools.evaluate_champion_audit import prepare as audit_prepare, run as audit_run, write

REFERENCE=ROOT/'results/champion_audit_v1_20261030/summary.json'
TOP2=ROOT/'controlled_experiments/original_top2_20261105'
OUTPUT=ROOT/'results/guide_on_native1280_test_v1_20261109'


def prepare(output):
    reference=json.loads(REFERENCE.read_text())
    original=reference['manifest'];settings=original['config']['settings']
    cfg=dict(version='guide-on-native1280-test-v1',data='datasets/pcb-native-res-cleaned-1280/data.yaml',
             expected_test_images=44,settings=settings,models=[],ensembles=[])
    old_agwd=next(m for m in original['config']['models'] if m['id']=='agwd_sdice')
    cfg['models'].append(dict(id='historical_agwd',name='Historical guide AGWD + Dice',
                              path=old_agwd['path'],sha256=old_agwd['sha256']))
    old_exp7=Path('/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen/runs/detect/runs/yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt')
    cfg['models'].append(dict(id='historical_exp7',name='Historical guide Exp7 / AS-PGMS',
                              path=str(old_exp7),sha256=file_hash(str(old_exp7))))
    identities={};source_reports={}
    for path in sorted((TOP2/'validation').glob('*.json')):
        r=json.loads(path.read_text());model='agwd' if r['model'].startswith('agwd') else 'exp7'
        key=f"{model}_640trained_s{r['seed']}"
        if file_hash(r['weights'])!=r['sha256']:
            raise ValueError('New guide checkpoint changed since training')
        cfg['models'].append(dict(id=key,name=f"{model.upper()} original-label 640-trained, seed {r['seed']}",
                                  path=r['weights'],sha256=r['sha256']))
        identities[key]=dict(training_input=640,training_dataset='info17-original-fourclass-v1',
                             model=r['model'],seed=r['seed'],source_report=str(path))
        source_reports[str(path)]=file_hash(str(path))
    if len(cfg['models'])!=8 or set(identities)!={f'{m}_640trained_s{s}' for m in ['agwd','exp7'] for s in [0,1,2]}:
        raise ValueError('Require both historical checkpoints and all six retrained guide checkpoints')
    pairs=[('historical_pair',['historical_agwd','historical_exp7'])]
    pairs += [(f'pair_640trained_s{s}',[f'agwd_640trained_s{s}',f'exp7_640trained_s{s}']) for s in [0,1,2]]
    cfg['ensembles']=[dict(id=k,name=k,members=members,weights=[1.30,1.25]) for k,members in pairs]
    config=ROOT/'configs/guide_native1280_comparison.json'
    if output.exists():raise FileExistsError(f'Already frozen: {output}; use run')
    write(config,cfg);audit_prepare(config,output)
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['test_fingerprint']!=original['test_fingerprint'] or manifest['records']!=original['records']:
        raise ValueError('Inputs differ from the original 68.74% reference audit')
    manifest['source_hashes'].update(source_reports)
    for path in [Path(__file__),ROOT/'models/as_pgms_yolo.py',ROOT/'tools/group_multistage_loss.py',ROOT/'tools/scale_adaptive_nwd_loss.py',REFERENCE]:
        manifest['source_hashes'][str(path)]=file_hash(str(path))
    manifest['comparison_reference']=dict(path=str(REFERENCE),sha256=file_hash(str(REFERENCE)),
        manifest_id=original['id'],test_fingerprint=original['test_fingerprint'],
        ensemble4=next(r['accuracy'] for r in reference['results'] if r['id']=='ensemble4'),
        historical_agwd=next(r['accuracy'] for r in reference['results'] if r['id']=='agwd_sdice'))
    manifest['new_training_identities']=identities
    manifest['disclosure']='Retrospective legacy test comparison. The six new checkpoints were trained at 640 on original labels; evaluation here is at 1280 on legacy main labels. Training recipes and prior held-out exposure differ. No test-based selection or weight search; this is not the original-label paper benchmark.'
    manifest.pop('id');manifest['id']=digest(manifest)
    write(output/'manifest.json',manifest)
    print('Frozen eight singles and four fixed WBF pairs against the exact 68.74% benchmark',flush=True)


def run(output,device):
    # Import the custom pickle implementation before any model deserialization.
    from models import as_pgms_yolo
    audit_run(output,device)
    raw=json.loads((output/'summary.json').read_text());manifest=raw['manifest']
    baseline=manifest['comparison_reference']['ensemble4']
    reproduced=next(r['accuracy'] for r in raw['results'] if r['id']=='historical_agwd')
    previous=manifest['comparison_reference']['historical_agwd']
    if any(abs(reproduced[k]-previous[k])>5e-4 for k in ['mAP50','mAP50_95']):
        raise ValueError('Historical AGWD control did not reproduce; investigate before comparing rankings')
    groups=[]
    for group,prefix in [('AGWD + Dice, 640-trained','agwd_640trained_s'),
                         ('Exp7 / AS-PGMS, 640-trained','exp7_640trained_s'),
                         ('Two-model WBF, 640-trained','pair_640trained_s')]:
        rows=[next(r for r in raw['results'] if r['id']==prefix+str(seed)) for seed in [0,1,2]]
        metrics={key:dict(mean_percent=statistics.mean(r['accuracy'][key]*100 for r in rows),
                         sample_sd_percent=statistics.stdev(r['accuracy'][key]*100 for r in rows),
                         per_seed_percent=[r['accuracy'][key]*100 for r in rows]) for key in ['mAP50','mAP50_95']}
        groups.append(dict(name=group,metrics=metrics))
    comparison=dict(status='completed_verified',reference_ensemble_mAP50_percent=baseline['mAP50']*100,
        reference_ensemble_mAP50_95_percent=baseline['mAP50_95']*100,groups=groups,
        historical=[r for r in raw['results'] if r['id'].startswith('historical_')],
        disclosure=manifest['disclosure'],same_test_fingerprint=True,
        historical_agwd_control_reproduced=True,results_file=str(output/'summary.json'))
    write(output/'comparison.json',comparison)
    print(json.dumps(comparison['groups'],indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--device',default='0');args=p.parse_args()
    if args.action=='prepare':prepare(args.output.resolve())
    else:run(args.output.resolve(),args.device)


if __name__=='__main__':main()
