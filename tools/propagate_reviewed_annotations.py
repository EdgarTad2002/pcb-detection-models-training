#!/usr/bin/env python3
"""Transfer explicit visual-review decisions to registered variants; never classify from confidence."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np


def features(path, sift):
    im=cv2.imread(str(path))
    if im is None:raise ValueError('Cannot decode '+str(path))
    scale=min(1.,1200/max(im.shape[:2]));im=cv2.resize(im,None,fx=scale,fy=scale)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);kp,des=sift.detectAndCompute(gray,None)
    return kp,des,scale


def registration(a,b):
    ka,da,sa=a;kb,db,sb=b
    matches=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
    if len(good)<40:raise ValueError('Insufficient feature matches')
    pa=np.float32([ka[m.queryIdx].pt for m in good]);pb=np.float32([kb[m.trainIdx].pt for m in good])
    H,mask=cv2.findHomography(pa,pb,cv2.RANSAC,2.)
    if H is None or int(mask.sum())<40 or mask.mean()<.5:raise ValueError('Registration failed')
    error=float(np.median(np.linalg.norm(cv2.perspectiveTransform(pa.reshape(-1,1,2),H).reshape(-1,2)[mask.ravel()>0]-pb[mask.ravel()>0],axis=1)))
    if error>.8:raise ValueError('Registration error too large')
    return np.diag([1/sb,1/sb,1])@H@np.diag([sa,sa,1]),dict(inliers=int(mask.sum()),median_error_resized_px=error)


def bounds(box,w=1,h=1):
    x,y,bw,bh=box
    return np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])


def overlap(a,b):
    inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(inter/(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter+1e-12))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--decisions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cv2.setNumThreads(2);sift=cv2.SIFT_create(nfeatures=4000)
    inv=json.loads(a.inventory.read_text());decisions=json.loads(a.decisions.read_text());records={r['id']:r for r in inv['records']};groups=defaultdict(list);by_ref=defaultdict(list)
    for r in inv['records']:groups[r['board_prefix']].append(r)
    for decision in decisions:by_ref[decision['id']].append(decision)
    patches=[];pending=[];used=set();cache={}
    for ref_id,ds in by_ref.items():
        ref=records[ref_id]
        for target in groups[ref['board_prefix']]:
            try:
                if target['id']==ref_id:H=np.eye(3);evidence=dict(inliers=None,median_error_resized_px=0.)
                else:
                    for row in [ref,target]:
                        if row['id'] not in cache:cache[row['id']]=features(row['image'],sift)
                    H,evidence=registration(cache[ref_id],cache[target['id']])
            except Exception as exc:
                pending.append(dict(reference=ref_id,target=target['id'],reason=str(exc)));continue
            for decision in ds:
                source_box=next(b for b in ref['boxes'] if b['line']==decision['line']);x1,y1,x2,y2=bounds(source_box['xywh'],ref['width'],ref['height'])
                points=cv2.perspectiveTransform(np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]).reshape(-1,1,2),H).reshape(-1,2)
                expected=np.r_[points.min(0),points.max(0)]/np.array([target['width'],target['height']]*2)
                expected=np.clip(expected,0.,1.)
                if np.any(expected[2:]<=expected[:2]):
                    continue  # The component is outside this cropped/rotated variant.
                ranked=sorted([(overlap(expected,bounds(b['xywh'])),b) for b in target['boxes'] if b['cls']==source_box['cls']],key=lambda pair:pair[0],reverse=True)
                if not ranked or ranked[0][0]<.55 or (len(ranked)>1 and ranked[0][0]-ranked[1][0]<.20):
                    pending.append(dict(reference=ref_id,target=target['id'],line=decision['line'],reason='No unique spatially matching annotation'));continue
                iou,matched=ranked[0];key=(target['id'],matched['line'])
                if key in used:raise ValueError('Conflicting transferred decisions')
                used.add(key)
                patch=dict(id=target['id'],action=decision['action'],line=matched['line'],expected=matched['raw'],reason=decision['reason'],evidence=decision['evidence'],reference_image=ref_id,reference_line=decision['line'],matched_annotation_iou=iou,registration=evidence)
                if decision['action']=='replace':patch['replacement']=str(decision['new_class'])+' '+' '.join(matched['raw'].split()[1:])
                patches.append(patch)
        print(ref['board_prefix'],len(ds),'reference decisions processed',flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(patches=patches,pending=pending),indent=2)+'\n');print('Transferred',len(patches),'pending',len(pending),flush=True)


if __name__=='__main__':main()
