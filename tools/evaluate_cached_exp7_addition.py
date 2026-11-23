#!/usr/bin/env python3
"""Fixed five-member WBF comparison using existing predictions; CPU scoring only."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.controlled_experiments import digest, file_hash, split_signature
from tools.evaluate_champion_audit import fuse, score, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/champion_plus_exp7_v1_20261110')
    args=p.parse_args();output=args.output.resolve()
    if output.exists():raise FileExistsError(f'Refuse to overwrite: {output}')
    import torch
    torch.set_num_threads(2)
    native=ROOT/'results/champion_audit_v1_20261030'
    guide=ROOT/'results/guide_on_native1280_test_v1_20261109'
    a=json.loads((native/'summary.json').read_text())
    b=json.loads((guide/'summary.json').read_text())
    m=a['manifest'];gm=b['manifest'];settings=m['config']['settings']
    if m['test_fingerprint']!=gm['test_fingerprint'] or m['records']!=gm['records']:
        raise ValueError('Different evaluation images or annotations')
    if settings!=gm['config']['settings']:raise ValueError('Different prediction/fusion settings')
    for package,version in m['versions'].items():
        if importlib.metadata.version(package)!=version:raise ValueError(f'Use frozen {package}=={version}')
    records=m['records'];sources={}
    for r in records:
        if file_hash(r['image'])!=r['image_sha256'] or file_hash(r['label'])!=r['label_sha256']:
            raise ValueError('Benchmark images or labels changed')
    if split_signature(records)!=m['test_fingerprint']:raise ValueError('Benchmark membership changed')
    members=['yolo12','crop','agwd_sdice','physot','historical_exp7']
    weights=[1.3,1.2,1.1,1.0,1.25]
    caches={};shape_reference=None
    for key in members+['ensemble4']:
        path=(guide if key=='historical_exp7' else native)/'predictions'/(key+'.json')
        c=json.loads(path.read_text());expected=gm['id'] if key=='historical_exp7' else m['id']
        if c['manifest_id']!=expected or c['image_names']!=[r['name'] for r in records]:
            raise ValueError('Prediction identity or image order mismatch')
        if shape_reference is None:shape_reference=c['shapes']
        if shape_reference!=c['shapes']:raise ValueError('Prediction coordinate scales differ')
        caches[key]=c['predictions'];sources[str(path)]=file_hash(str(path))
    baseline=next(r['accuracy'] for r in a['results'] if r['id']=='ensemble4')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),settings=settings,
                  members=members,weights=weights,test_fingerprint=m['test_fingerprint'],
                  reference_manifest=m['id'],guide_manifest=gm['id'],source_hashes=sources,
                  source_code_sha256=file_hash(str(Path(__file__))),
                  method='Cached predictions; exact version-pinned WBF and AP scorer; no model inference or timing.',
                  selection='One prespecified extra-member weight (1.25) from the guide. No test-set weight search.',
                  disclosure='Historical legacy benchmark with known label and split limitations; not a new independent holdout.')
    manifest['id']=digest(manifest);write(output/'manifest.json',manifest)
    # Validate both scoring and local fusion before measuring the new configuration.
    control=score(caches['ensemble4'],records,shape_reference,settings)
    rebuilt=[];extended=[]
    for i,shape in enumerate(shape_reference):
        rebuilt.append(fuse([caches[k][i] for k in members[:4]],shape,weights[:4],settings))
        extended.append(fuse([caches[k][i] for k in members],shape,weights,settings))
        if (i+1)%11==0:print(f'Fused {i+1}/{len(records)} images',flush=True)
    rebuilt_metrics=score(rebuilt,records,shape_reference,settings)
    for result in [control,rebuilt_metrics]:
        for metric in ['mAP50','mAP50_95']:
            if abs(result[metric]-baseline[metric])>2e-6:
                raise ValueError('Existing ensemble does not reproduce; do not compare')
        for name,expected in baseline['per_class'].items():
            if abs(result['per_class'][name]['ap50']-expected['ap50'])>2e-6:
                raise ValueError('Existing per-class scores do not reproduce')
    five=score(extended,records,shape_reference,settings)
    comparison=dict(status='completed_verified',baseline=baseline,five_model=five,
                    mAP50_change_percentage_points=100*(five['mAP50']-baseline['mAP50']),
                    mAP50_95_change_percentage_points=100*(five['mAP50_95']-baseline['mAP50_95']),
                    baseline_cached_score_reproduced=True,baseline_four_member_fusion_reproduced=True,
                    manifest=manifest)
    write(output/'predictions.json',dict(manifest_id=manifest['id'],image_names=[r['name'] for r in records],
                                       shapes=shape_reference,predictions=extended))
    write(output/'summary.json',comparison)
    print(json.dumps({k:comparison[k] for k in ['status','mAP50_change_percentage_points','mAP50_95_change_percentage_points']}),flush=True)
    print(f"Four-model AP50: {baseline['mAP50']*100:.4f}%; five-model AP50: {five['mAP50']*100:.4f}%",flush=True)


if __name__=='__main__':main()
