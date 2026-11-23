#!/usr/bin/env python3
"""Inventory every native PCB image/label, render review pages, and apply traced edits to a new version."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

NAMES = ['Capacitor', 'Connector', 'Electrolytic Capacitor', 'IC']
COLORS = ['#45f5a0', '#ff65de', '#ffc447', '#57b8ff']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def read_boxes(path):
    boxes, issues = [], []
    if not path.exists():
        return [], [dict(kind='missing_label')]
    for line_number, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        try:
            vals = [float(x) for x in text.split()]
            if len(vals) != 5 or not all(math.isfinite(x) for x in vals):
                raise ValueError('Expected five finite numbers')
            cls, cx, cy, w, h = vals
            if cls != int(cls) or int(cls) not in range(4):
                raise ValueError('Invalid class')
            if w <= 0 or h <= 0:
                raise ValueError('Nonpositive box')
            boxes.append(dict(line=line_number, cls=int(cls), xywh=[cx, cy, w, h], raw=text))
            if min(cx-w/2, cy-h/2) < -1e-6 or max(cx+w/2, cy+h/2) > 1+1e-6:
                issues.append(dict(kind='outside_image', line=line_number, xywh=[cx, cy, w, h]))
        except ValueError as exc:
            issues.append(dict(kind='invalid_row', line=line_number, raw=text, reason=str(exc)))
    return boxes, issues


def xyxy(b):
    x, y, w, h = b['xywh']
    return [x-w/2, y-h/2, x+w/2, y+h/2]


def inventory(source, output):
    if (output/'inventory.json').exists():
        raise FileExistsError('An inventory is immutable; use a fresh output directory')
    records, groups, files, pixels = [], defaultdict(list), defaultdict(list), defaultdict(list)
    for split in ['train', 'valid', 'test']:
        directory = source/split/'images'
        images = sorted(p for p in directory.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp'])
        for image in images:
            label = source/split/'labels'/(image.stem+'.txt')
            boxes, issues = read_boxes(label)
            record = dict(id=f'{split}/{image.name}', split=split, image=str(image.resolve()), label=str(label.resolve()),
                          image_sha256=sha(image), label_sha256=sha(label) if label.exists() else None, boxes=boxes, issues=issues,
                          board_prefix=image.name.split('.rf.')[0], visual_review='pending')
            try:
                with Image.open(image) as im:
                    rgb = im.convert('RGB'); rgb.load()
                    record.update(width=rgb.width, height=rgb.height, pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                    pixels[record['pixel_sha256']].append(record['id'])
                    a = np.asarray(ImageOps.grayscale(rgb).resize((64, 64)), dtype=np.float32)
                    import cv2
                    dct = cv2.dct(a)[:8, :8].ravel()[1:]
                    record['phash'] = hex(sum(int(v>np.median(dct)) << i for i,v in enumerate(dct)))
                for b in boxes:
                    if min(b['xywh'][2]*record['width'], b['xywh'][3]*record['height']) < 2:
                        issues.append(dict(kind='sub_2px_box', line=b['line']))
            except Exception as exc:
                issues.append(dict(kind='image_decode_error', reason=str(exc)))
            if boxes:
                a = np.asarray([xyxy(b) for b in boxes]); wh = np.maximum(a[:, 2:]-a[:, :2], 0); areas = wh.prod(axis=1)
                for i in range(len(boxes)):
                    intersection = np.maximum(np.minimum(a[i, 2:],a[i+1:, 2:])-np.maximum(a[i,:2],a[i+1:,:2]),0).prod(axis=1)
                    iou = intersection/(areas[i]+areas[i+1:]-intersection+1e-12)
                    for j in np.flatnonzero(iou>.85)+i+1:
                        same = boxes[i]['cls']==boxes[j]['cls']
                        exact = same and boxes[i]['xywh']==boxes[j]['xywh']
                        issues.append(dict(kind='exact_duplicate' if exact else 'high_overlap_same_class' if same else 'high_overlap_class_conflict', lines=[boxes[i]['line'],boxes[j]['line']], iou=float(iou[j-i-1])))
            records.append(record)
            groups[record['board_prefix']].append(record['id']); files[record['image_sha256']].append(record['id'])
        image_stems = {p.stem for p in images}
        orphan = sorted(p.name for p in (source/split/'labels').glob('*.txt') if p.stem not in image_stems)
        if orphan:
            save(output/f'{split}_orphan_labels.json',orphan)
    shared = {k:v for k,v in groups.items() if len({x.split('/')[0] for x in v})>1}
    summary = dict(images=len(records), boxes=sum(len(r['boxes']) for r in records),
                   splits={s:dict(images=sum(r['split']==s for r in records), boxes=sum(len(r['boxes']) for r in records if r['split']==s), classes=dict(Counter(NAMES[b['cls']] for r in records if r['split']==s for b in r['boxes']))) for s in ['train','valid','test']},
                   issue_counts=dict(Counter(i['kind'] for r in records for i in r['issues'])), unique_board_prefixes=len(groups),
                   cross_split_board_prefix_groups=shared, identical_file_groups={k:v for k,v in files.items() if len(v)>1},
                   identical_pixel_groups={k:v for k,v in pixels.items() if len(v)>1},
                   scope='Structural scan of every file; visual review status tracked separately. Filename and perceptual similarities are review candidates, not confirmed leakage.')
    save(output/'inventory.json',dict(source=str(source.resolve()),records=records,summary=summary))
    save(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,dict) or k in ['splits','issue_counts']},indent=2))


def overlay(record, size=(1000, 720), labels=True):
    with Image.open(record['image']) as im:
        image = im.convert('RGB')
    image.thumbnail(size)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 12)
    for b in record['boxes']:
        x1,y1,x2,y2=xyxy(b); rect=[x1*image.width,y1*image.height,x2*image.width,y2*image.height]
        draw.rectangle(rect,outline=COLORS[b['cls']],width=1)
        if labels:
            draw.text((rect[0],rect[1]-11),f"{b['line']}:{b['cls']}",font=font,fill=COLORS[b['cls']],stroke_width=1,stroke_fill='black')
    return image


def render(inventory_path, output, per_page=6):
    inv=json.loads(inventory_path.read_text());records=inv['records']
    output.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype(FONT,17)
    for start in range(0,len(records),per_page):
        page=Image.new('RGB',(2040,800*((per_page+1)//2)), '#162132'); draw=ImageDraw.Draw(page)
        for n,r in enumerate(records[start:start+per_page]):
            x=(n%2)*1020;y=(n//2)*800
            image=overlay(r,labels=False);page.paste(image,(x+(1000-image.width)//2,y+65))
            draw.text((x+12,y+8),f"{start+n:03d} {r['id'].split('/')[0]} {r['board_prefix'][:50]}",font=font,fill='white')
            draw.text((x+12,y+32),f"{r.get('width')}x{r.get('height')} | boxes={len(r['boxes'])} | issues={len(r['issues'])}",font=font,fill='white')
        page.save(output/f'page_{start//per_page:03d}.jpg',quality=92)
    print(f'Rendered {math.ceil(len(records)/per_page)} pages')


def apply(inventory_path, patches_path, destination, exclude_path=None):
    """Apply exact-line edits to copied labels, with source locks and explicit evidence."""
    inv=json.loads(inventory_path.read_text());patches=json.loads(patches_path.read_text())
    records={r['id']:r for r in inv['records']}
    excluded = json.loads(exclude_path.read_text()) if exclude_path else {}
    if any(key not in records for key in excluded):
        raise ValueError('Unknown quarantine image')
    if destination.exists():
        raise FileExistsError('Refuse to overwrite any dataset')
    grouped=defaultdict(list)
    for patch in patches:
        if patch['id'] not in records or not patch.get('reason') or not patch.get('evidence'):
            raise ValueError('Every edit requires an existing image, reason, and review evidence')
        if patch['action'] not in ['delete','replace','add']:
            raise ValueError('Unknown action')
        grouped[patch['id']].append(patch)
    # Validate the complete source and every patch before writing any output.
    prepared={}
    for key,r in records.items():
        if sha(r['image'])!=r['image_sha256'] or sha(r['label'])!=r['label_sha256']:
            raise ValueError('Source changed: '+key)
        lines=Path(r['label']).read_text().splitlines(); edits={}; additions=[]
        for p in grouped[key]:
            if p['action'] in ['delete','replace']:
                index=p['line']-1
                if index in edits or index<0 or index>=len(lines) or lines[index]!=p['expected']:
                    raise ValueError('Conflicting edit or mismatched source row')
                edits[index]=None if p['action']=='delete' else p['replacement']
            else:
                additions.append(p['replacement'])
        updated=[edits.get(i,line) for i,line in enumerate(lines) if edits.get(i,line) is not None]+additions
        for line in updated:
            v=[float(t) for t in line.split()]
            if len(v)!=5 or not all(math.isfinite(t) for t in v) or v[0]!=int(v[0]) or not 0<=v[0]<4 or min(v[3:])<=0:
                raise ValueError('Invalid output annotation')
            _,x,y,w,h=v
            if min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:
                raise ValueError('Output box outside image')
        prepared[key]=updated
    destination.mkdir(parents=True)
    for key,r in records.items():
        split=r['split'];base=destination/'quarantine'/split if key in excluded else destination/split
        im=base/'images'/Path(r['image']).name;label=base/'labels'/Path(r['label']).name
        im.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        im.symlink_to(r['image']);label.write_text('\n'.join(prepared[key])+'\n' if prepared[key] else '')
    (destination/'data.yaml').write_text(f"path: {destination.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 4\nnames: {json.dumps(NAMES)}\n")
    save(destination/'correction_manifest.json',dict(source_inventory=str(inventory_path.resolve()),source_inventory_sha256=sha(inventory_path),patches=patches,images=len(records),quarantine=excluded,status='review_candidate_not_frozen_for_paper',split_policy='Original split memberships preserved for included images; explicitly listed lower-priority overlapping groups are quarantined. No test images or labels moved into training.'))
    print(destination)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('inventory');a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('render');a.add_argument('--inventory',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--per-page',type=int,default=6)
    a=sub.add_parser('apply');a.add_argument('--inventory',type=Path,required=True);a.add_argument('--patches',type=Path,required=True);a.add_argument('--destination',type=Path,required=True);a.add_argument('--exclude',type=Path)
    args=p.parse_args()
    if args.action=='inventory':inventory(args.source,args.output)
    elif args.action=='render':render(args.inventory,args.output,args.per_page)
    else:apply(args.inventory,args.patches,args.destination,args.exclude)


if __name__=='__main__':
    main()
