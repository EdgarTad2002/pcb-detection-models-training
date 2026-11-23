#!/usr/bin/env python3
"""Frozen, original-label 640-pixel retraining of AGWD-Dice and checkpoint-confirmed AS-PGMS exp7."""
import argparse, hashlib, importlib.metadata, json, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def prepare(out):
 from tools.evaluate_info17_protocol import audit_dataset
 if out.exists():raise FileExistsError(out)
 dataset=audit_dataset(ROOT/'datasets/info17-original-fourclass-v1')
 old=json.loads((ROOT/'controlled_experiments/751837c92ab4/plan.json').read_text())
 template=next(x['command'] for x in old['jobs'] if x['model']=='yolo26s_agwd_sdice' and x['seed']==0)
 out.mkdir(parents=True)
 source=out/'source_snapshot'
 paths=list(ROOT.glob('*.py'))+list((ROOT/'tools').glob('*.py'))+[ROOT/'models/as_pgms_yolo.py',ROOT/'data/pcb_spectral_priors.json', ROOT/'configs/original_top2_source.json']
 hashes={}
 for p in paths:
  rel=p.relative_to(ROOT);target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);hashes[str(rel)]=sha(target)
 jobs=[]
 for model in ['agwd_sdice_smooth001_uniform','exp7_as_pgms']:
  for seed in [0,1,2]:
   key=f'original_top2_20261105_{model}_640_s{seed}'
   if (ROOT/'runs'/key).exists():raise FileExistsError(key)
   command=list(template)
   for option,value in {'--data':str(ROOT/'datasets/info17-original-fourclass-v1/data.yaml'),'--eval-data':str(ROOT/'datasets/info17-original-fourclass-v1/data.yaml'),'--run-key':key,'--seed':str(seed),'--results-dir':str(out/'training_validation'),'--weights':str(ROOT/'yolo26s.pt')}.items():command[command.index(option)+1]=value
   command[0]=sys.executable;command[1]=str(source/'train.py')
   jobs.append(dict(model=model,seed=seed,key=key,agwd_command=command))
 manifest=dict(root=str(ROOT),output=str(out),sources=hashes,initializer={'path':str(ROOT/'yolo26s.pt'),'sha256':sha(ROOT/'yolo26s.pt')},dataset=dataset,jobs=jobs,epochs=100,imgsz=640,batch=16,nbs=16,selection='best.pt by validation only; no automatic test evaluation or test-based tuning',exp7=dict(lambda_sr=.10,lambda_group=.20,smooth=.0001,stage_weights=[.4,.3,.15,.1],box=5.,cls=1.5,dfl=2.,historical_auxiliary_settings='Not recoverable from stripped checkpoint; explicitly adopt checkpoint-compatible implementation defaults.'))
 manifest['environment']={k:importlib.metadata.version(k) for k in ['ultralytics','torch']}
 yaml=ROOT/'datasets/info17-original-fourclass-v1/data.yaml'
 manifest['data_yaml']={'path':str(yaml),'sha256':sha(yaml)}
 write(out/'plan.json',manifest);print(out/'plan.json',flush=True)
def verify(plan):
 out=Path(plan['output'])
 for rel,h in plan['sources'].items():
  if sha(out/'source_snapshot'/rel)!=h:raise ValueError('Frozen source changed: '+rel)
 if sha(plan['initializer']['path'])!=plan['initializer']['sha256']:raise ValueError('Initializer changed')
 if sha(plan['data_yaml']['path'])!=plan['data_yaml']['sha256']:raise ValueError('Dataset YAML changed')
 for package,version in plan['environment'].items():
  if importlib.metadata.version(package)!=version:raise ValueError('Environment changed: '+package)
 for split in plan['dataset']['splits'].values():
  for r in split['records']:
   if sha(r['image'])!=r['image_sha256'] or sha(r['label'])!=r['label_sha256']:raise ValueError('Dataset changed')
def exp7_trainer(plan,job,device,extra_overrides=None):
 from models.as_pgms_yolo import ASPGMSTrainer
 ASPGMSTrainer.lambda_sr=.10;ASPGMSTrainer.lambda_group=.20;ASPGMSTrainer.use_squared_dice=True
 overrides=dict(model=plan['initializer']['path'],data=str(Path(plan['root'])/'datasets/info17-original-fourclass-v1/data.yaml'),project=str(Path(plan['root'])/'runs'/job['key']),name='pcb-filtered',exist_ok=False,epochs=100,imgsz=640,batch=16,nbs=16,workers=8,device=device,optimizer='SGD',lr0=.01,lrf=.01,momentum=.937,weight_decay=.0005,warmup_epochs=3.,patience=0,seed=job['seed'],deterministic=True,cache=False,amp=True,plots=True,save=True,box=5.,cls=1.5,dfl=2.,copy_paste=0.,mixup=0.,mosaic=1.,close_mosaic=10,conf=.001,iou=.5,max_det=1000)
 overrides.update(extra_overrides or {})
 return ASPGMSTrainer(overrides=overrides)
def smoke(plan):
 sys.path.insert(0,str(Path(plan['output'])/'source_snapshot'))
 import torch
 from types import SimpleNamespace
 from ultralytics import YOLO
 from models.as_pgms_yolo import ASPGMSDetectionModel
 from tools.scale_adaptive_nwd_loss import ScaleAdaptiveNWDBboxLoss
 torch.set_num_threads(2)
 base=YOLO(plan['initializer']['path'])
 assert base.model.model[-1].nc == 80, 'Expected COCO initializer, not a PCB checkpoint'
 model=ASPGMSDetectionModel(cfg=base.model.yaml,nc=4,verbose=False,imgsz=640)
 model.load(base.model,verbose=False);model.inject_physics_and_micro_bridges()
 model.args=SimpleNamespace(box=5.,cls=1.5,dfl=2.,epochs=100)
 model.train()
 b={'img':torch.rand(2,3,64,64),'cls':torch.tensor([[0.],[3.]]),'bboxes':torch.tensor([[.5,.5,.15,.15],[.5,.5,.3,.3]]),'batch_idx':torch.tensor([0.,1.])}
 loss,_=model.loss(b);loss.sum().backward()
 assert torch.isfinite(loss).all()
 assert model.mask_head is not None and model.sr_head is not None
 for head in [model.mask_head,model.sr_head]:
  assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in head.parameters())
 criterion=model.criterion
 assert all(isinstance(getattr(criterion,k).bbox_loss,ScaleAdaptiveNWDBboxLoss) for k in ['one2many','one2one'])
 # Exercise the actual trainer's save/strip/load path on disposable synthetic data.
 from PIL import Image
 smoke_dir=Path(plan['output'])/'smoke_trainer'
 if smoke_dir.exists():raise FileExistsError(smoke_dir)
 for split in ['train','val']:
  (smoke_dir/split/'images').mkdir(parents=True)
  (smoke_dir/split/'labels').mkdir()
  for i in range(2):
   Image.new('RGB',(128,128),(50+i*70,100,150)).save(smoke_dir/split/'images'/f'{i}.png')
   (smoke_dir/split/'labels'/f'{i}.txt').write_text(f'{i*3} 0.5 0.5 0.3 0.3\n')
 data=smoke_dir/'data.yaml'
 data.write_text(f'path: {smoke_dir}\ntrain: train/images\nval: val/images\nnames: [Capacitor, Connector, Electrolytic Capacitor, IC]\n')
 overrides=dict(data=str(data),project=str(smoke_dir/'runs'),name='save_reload',epochs=1,imgsz=128,batch=2,nbs=2,workers=0,device='cpu',amp=False,plots=False,mosaic=0.,close_mosaic=0,save=True)
 trainer=exp7_trainer(plan,{'key':'smoke_unused','seed':0},'cpu',overrides);trainer.train()
 saved=Path(trainer.save_dir)/'weights/best.pt'
 reloaded=YOLO(str(saved))
 result=reloaded.predict(torch.rand(1,3,128,128),device='cpu',verbose=False)
 assert len(result)==1
 write(Path(plan['output'])/'smoke.json',dict(status='passed',loss=float(loss.sum().detach()),backward=True,auxiliary_mask_gradient=True,auxiliary_sr_gradient=True,nwd_wrappers=True,trainer_save_reload=True,synthetic_data_only=True))
 print('AS-PGMS forward/backward smoke passed',flush=True)
def run(plan,index,device):
 verify(plan);job=plan['jobs'][index]
 sys.path.insert(0,str(Path(plan['output'])/'source_snapshot'))
 if (Path(plan['root'])/'runs'/job['key']).exists():raise FileExistsError('Refuse reuse existing run')
 if job['model'].startswith('agwd'):
  command=job['agwd_command'];command[command.index('--device')+1]=device
  subprocess.run(command,cwd=plan['root'],check=True)
 else:
  trainer=exp7_trainer(plan,job,device);trainer.train()
 from ultralytics import YOLO
 from models import as_pgms_yolo  # Register checkpoint classes for either method.
 weights=Path(plan['root'])/'runs'/job['key']/'pcb-filtered/weights/best.pt'
 if not weights.is_file():raise FileNotFoundError(weights)
 metrics=YOLO(str(weights)).val(data=plan['data_yaml']['path'],split='val',imgsz=640,rect=False,batch=1,device=device,conf=.001,iou=.5,max_det=1000,half=False,plots=False,project=str(Path(plan['output'])/'evaluation_artifacts'),name=job['key'],exist_ok=False)
 write(Path(plan['output'])/'validation'/f"{job['key']}.json",dict(model=job['model'],seed=job['seed'],weights=str(weights),sha256=sha(weights),split='val',protocol=dict(imgsz=640,rect=False,batch=1,conf=.001,iou=.5,max_det=1000,half=False),mAP50=float(metrics.box.map50),mAP50_95=float(metrics.box.map),precision=float(metrics.box.mp),recall=float(metrics.box.mr),per_class=[dict(index=int(i),name=metrics.names[int(i)],ap50=float(a),ap50_95=float(b),precision=float(p),recall=float(r)) for i,a,b,p,r in zip(metrics.box.ap_class_index,metrics.box.ap50,metrics.box.ap,metrics.box.p,metrics.box.r)]))
 write(Path(plan['output'])/'completed'/f"{job['key']}.json",dict(status='training_completed',job=job,selection='validation only'))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','smoke','run']);p.add_argument('--output',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--index',type=int);p.add_argument('--device',default='0');a=p.parse_args()
 if a.action=='prepare':
  if a.output is None:p.error('prepare requires --output')
  prepare(a.output.resolve());return
 if a.plan is None:p.error('smoke/run require --plan')
 plan=json.loads(a.plan.read_text());verify(plan)
 if a.action=='smoke':smoke(plan)
 else:
  if a.index is None or not 0<=a.index<len(plan['jobs']):p.error('run requires a valid --index')
  if not (Path(plan['output'])/'smoke.json').is_file():raise RuntimeError('Run smoke before training')
  run(plan,a.index,a.device)
if __name__=='__main__':main()
