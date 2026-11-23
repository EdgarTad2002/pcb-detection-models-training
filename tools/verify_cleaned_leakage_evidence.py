#!/usr/bin/env python3
"""Recheck all exports in the three named alias groups; never edit datasets."""
import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deep_audit_cleaned_dataset import features, register, save

OUT = Path('data_review/cleaned1280_deep_audit_20261115')
CACHE = Path('/tmp/pcb-cleaned-leakage-features-20261115')


def gray(row, flip=False):
    g = cv2.imread(row['image'], 0)
    s = min(1., 1100 / max(g.shape))
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(np.fliplr(g)) if flip else g


def flipped_features(row):
    g = gray(row, True)
    kp, des = cv2.SIFT_create(nfeatures=1800).detectAndCompute(g, None)
    return dict(points=np.float32([p.pt for p in kp]), descriptors=des,
                shape=list(g.shape))


def photometric(a, b, H, flip):
    ga, gb = gray(a, flip), gray(b)
    H = np.asarray(H)
    warped = cv2.warpPerspective(ga, H, (gb.shape[1], gb.shape[0]))
    mask = cv2.warpPerspective(np.ones_like(ga), H, (gb.shape[1], gb.shape[0]))
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8)).astype(bool)
    hp_a = warped.astype(float) - cv2.GaussianBlur(warped, (0, 0), 2).astype(float)
    hp_b = gb.astype(float) - cv2.GaussianBlur(gb, (0, 0), 2).astype(float)
    mask &= np.abs(hp_b) > 3
    return dict(overlap_texture_pixels=int(mask.sum()),
                highpass_correlation=float(np.corrcoef(hp_a[mask], hp_b[mask])[0, 1]),
                gray_correlation=float(np.corrcoef(warped[mask], gb[mask])[0, 1]))


def main():
    rows = json.loads((OUT / 'inventory.json').read_text())['records']
    groups = defaultdict(list)
    names = {'arduinomegatop', 'duetwifibottom', 'duetwifitop'}
    for r in rows:
        if r['normalized_view'] in names:
            groups[r['normalized_view']].append(r)
    ft = {r['id']: features(r, CACHE) for group in groups.values() for r in group}
    flipped = {}
    results = []
    representatives = []
    for name, group in sorted(groups.items()):
        pairs = []
        for a in group:
            if a['split'] != 'train':
                continue
            for b in group:
                if b['split'] == 'train':
                    continue
                matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=128))
                matcher.add([ft[b['id']]['descriptors']]); matcher.train()
                options = []
                for mirror in [False, True]:
                    if mirror and a['id'] not in flipped:
                        flipped[a['id']] = flipped_features(a)
                    hit = register(flipped[a['id']] if mirror else ft[a['id']], ft[b['id']], matcher)
                    if hit:
                        options.append((hit, mirror))
                if not options:
                    pairs.append(dict(a=a['id'], b=b['id'], status='no_registration'))
                    continue
                hit, mirror = max(options, key=lambda x: x[0]['inliers'])
                photo = photometric(a, b, hit['H'], mirror)
                hit.update(a=a['id'], b=b['id'], horizontal_flip_a=mirror, **photo)
                hit['same_photo_geometry_and_texture'] = (hit['inliers'] >= 100 and
                    min(hit['coverage_a'], hit['coverage_b']) > .10 and
                    hit['highpass_correlation'] > .80)
                pairs.append(hit)
        good = [p for p in pairs if p.get('same_photo_geometry_and_texture')]
        endpoints = sorted({p[k] for p in good for k in ['a','b']})
        results.append(dict(group=name, pairs=pairs, all_export_ids=[r['id'] for r in group],
                            verified_endpoint_ids=endpoints))
        best = max(good, key=lambda x: x['inliers'])
        representatives.append((name, next(r for r in group if r['id']==best['a']),
                                next(r for r in group if r['id']==best['b']), best))
    save(OUT / 'known_alias_reverification.json', results)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    canvas = Image.new('RGB', (1500, 540*len(representatives)), '#eef2f7')
    draw = ImageDraw.Draw(canvas)
    for i, (name, a, b, hit) in enumerate(representatives):
        y = i*540
        draw.text((20,y+8),f'{name}: SAME PHOTOGRAPH across splits',font=font,fill='black')
        draw.text((20,y+42),f"{hit['inliers']} geometric inliers; texture correlation {hit['highpass_correlation']:.3f}",font=font,fill='black')
        for j, row in enumerate([a,b]):
            im = Image.open(row['image']).convert('RGB'); im.thumbnail((720,420))
            x = j*750 + (750-im.width)//2
            canvas.paste(im,(x,y+110))
            draw.text((j*750+20,y+78),f"{row['split'].upper()} - {row['board_prefix']}",font=font,fill='black')
    canvas.save(OUT / 'confirmed_leakage_evidence.jpg',quality=93)
    print(json.dumps([dict(group=g['group'], pairs=len(g['pairs']),
        confirmed_pairs=sum(p.get('same_photo_geometry_and_texture',False) for p in g['pairs']),
        verified_endpoints=len(g['verified_endpoint_ids']),
        min_texture_correlation=min(p.get('highpass_correlation',1) for p in g['pairs'])) for g in results],indent=2))


if __name__ == '__main__':
    main()
