#!/usr/bin/env python3
"""Read-only PCB split audit: hashes, labels, exhaustive view-level SIFT registration.

No dataset files are changed. Similarity candidates require visual adjudication;
neither filename overlap nor feature matching proves a shared physical specimen.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from audit_native_dataset import read_boxes

cv2.setNumThreads(1)
cv2.setRNGSeed(417)
SPLITS = ['train', 'valid', 'test']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def normalized(name, family=False):
    value = name.split('.rf.')[0].lower()
    value = re.sub(r'([_.-](?:jpg|jpeg|png|bmp))+$', '', value)
    if family:
        value = re.sub(r'[_ -](?:top|bottom)$', '', value)
    return re.sub(r'[^a-z0-9]', '', value)


def phashes(rgb):
    gray = np.asarray(ImageOps.grayscale(rgb).resize((64, 64)), np.float32)
    values = []
    for a in [gray, np.fliplr(gray)]:
        for k in range(4):
            dct = cv2.dct(np.ascontiguousarray(np.rot90(a, k)))[:8, :8].ravel()[1:]
            values.append(sum(int(v > np.median(dct)) << i for i, v in enumerate(dct)))
    return values


def inventory(args):
    args.output.mkdir(parents=True, exist_ok=False)
    old = json.loads(args.previous.read_text())
    oldmap = {r['id']: r for r in old['records']}
    manifest = json.loads(args.champion_manifest.read_text())
    rows = []
    for split in SPLITS:
        paths = sorted(p for p in (args.source / split / 'images').iterdir()
                       if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        for path in paths:
            label = args.source / split / 'labels' / (path.stem + '.txt')
            key = split + '/' + path.name
            boxes, issues = read_boxes(label)
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                w, h = rgb.size
                px = hashlib.sha256(f'{w}x{h}:'.encode() + rgb.tobytes()).hexdigest()
                hashes = phashes(rgb)
            image_sha, label_sha = digest(path), digest(label)
            previous = oldmap.get(key, {})
            for b in boxes:
                if min(b['xywh'][2] * w, b['xywh'][3] * h) < 2:
                    issues.append({'kind': 'sub_2px_box', 'line': b['line']})
            arr = np.asarray([b['xywh'] for b in boxes], float).reshape(-1, 4)
            if len(arr):
                xyxy = np.column_stack([arr[:, :2] - arr[:, 2:] / 2,
                                        arr[:, :2] + arr[:, 2:] / 2])
                area = arr[:, 2:].prod(1)
                for i in range(len(arr)):
                    inter = np.maximum(np.minimum(xyxy[i, 2:], xyxy[i+1:, 2:]) -
                                       np.maximum(xyxy[i, :2], xyxy[i+1:, :2]), 0).prod(1)
                    overlaps = inter / (area[i] + area[i+1:] - inter + 1e-12)
                    for j in np.flatnonzero(overlaps > .85) + i + 1:
                        issues.append({'kind': 'overlap_above_085',
                                       'lines': [boxes[i]['line'], boxes[j]['line']]})
            rows.append(dict(id=key, split=split, image=str(path.resolve()),
                             label=str(label.resolve()), image_sha256=image_sha,
                             label_sha256=label_sha, pixel_sha256=px, width=w, height=h,
                             board_prefix=path.name.split('.rf.')[0],
                             normalized_view=normalized(path.name),
                             normalized_family=normalized(path.name, True),
                             phashes=hashes, classes=dict(Counter(b['cls'] for b in boxes)),
                             boxes=len(boxes), issues=issues,
                             previous_hashes_match=image_sha == previous.get('image_sha256')
                             and label_sha == previous.get('label_sha256')))
        print('Inventoried', split, len(paths), flush=True)
    byname = {Path(r['image']).name: r for r in rows if r['split'] == 'test'}
    champion_match = all(r['name'] in byname and
                         byname[r['name']]['image_sha256'] == r['image_sha256'] and
                         byname[r['name']]['label_sha256'] == r['label_sha256']
                         for r in manifest['records'])
    active, revised_counts, revised_image_mismatches, revised_label_diffs = {}, {}, [], []
    for split in SPLITS:
        revised_counts[split] = Counter()
        for path in sorted((args.revised / split / 'images').iterdir()):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
            key = split + '/' + path.name
            active[key] = digest(path)
            label = args.revised / split / 'labels' / (path.stem + '.txt')
            boxes, _ = read_boxes(label)
            revised_counts[split].update(b['cls'] for b in boxes)
            original = next((r for r in rows if r['id'] == key), None)
            if original is None or active[key] != original['image_sha256']:
                revised_image_mismatches.append(key)
            if original and digest(label) != original['label_sha256']:
                revised_label_diffs.append(key)
    for row in rows:
        row['active_in_r4'] = row['id'] in active
    hashes = {}
    for field in ['image_sha256', 'pixel_sha256', 'board_prefix', 'normalized_view', 'normalized_family']:
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row['id'])
        hashes[field] = {k: v for k, v in groups.items()
                         if len({x.split('/')[0] for x in v}) > 1}
    summary = dict(source=str(args.source.resolve()), revised=str(args.revised.resolve()),
                   images=len(rows), boxes=sum(r['boxes'] for r in rows),
                   previous_inventory_matching=sum(r['previous_hashes_match'] for r in rows),
                   champion_test_records=len(manifest['records']), champion_test_hashes_match=champion_match,
                   splits={s: dict(images=sum(r['split'] == s for r in rows),
                       boxes=sum(r['boxes'] for r in rows if r['split'] == s),
                       classes=dict(sum((Counter(r['classes']) for r in rows if r['split'] == s), Counter())),
                       long_side_below_1280=sum(max(r['width'], r['height']) < 1280 for r in rows if r['split'] == s),
                       view_groups=len({r['board_prefix'] for r in rows if r['split'] == s})) for s in SPLITS},
                   issues=dict(Counter(i['kind'] for r in rows for i in r['issues'])),
                   cross_split_groups=hashes, revised_active=len(active),
                   revised_excluded=[r['id'] for r in rows if not r['active_in_r4']],
                   revised_image_mismatches=revised_image_mismatches,
                   revised_changed_label_files=revised_label_diffs,
                   revised_class_counts={s: dict(c) for s, c in revised_counts.items()})
    save(args.output / 'inventory.json', dict(records=rows, summary=summary))
    save(args.output / 'inventory_summary.json', summary)
    print('Preserved original hashes:', summary['previous_inventory_matching'],
          'Champion test matches:', champion_match, flush=True)


def features(row, cache):
    path = cache / (row['image_sha256'] + '.npz')
    if not path.exists():
        gray = cv2.imread(row['image'], cv2.IMREAD_GRAYSCALE)
        scale = min(1., 1100 / max(gray.shape))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        kp, des = cv2.SIFT_create(nfeatures=1800).detectAndCompute(gray, None)
        if des is None:
            des, points = np.empty((0, 128), np.uint8), np.empty((0, 2), np.float32)
        else:
            points = np.asarray([k.pt for k in kp], np.float32)
            des = des.astype(np.uint8)
        np.savez_compressed(path, points=points, descriptors=des, shape=gray.shape)
    z = np.load(path)
    return dict(points=z['points'], descriptors=z['descriptors'].astype(np.float32),
                shape=z['shape'].tolist())


def register(a, b, matcher):
    if min(len(a['points']), len(b['points'])) < 12:
        return None
    matches = matcher.knnMatch(a['descriptors'], k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair]
            if m.distance < .70 * n.distance]
    good = sorted(good, key=lambda m: m.distance)
    seen, unique = set(), []
    for m in good:
        if m.trainIdx not in seen:
            seen.add(m.trainIdx)
            unique.append(m)
    if len(unique) < 16:
        return None
    pa = np.float32([a['points'][m.queryIdx] for m in unique])
    pb = np.float32([b['points'][m.trainIdx] for m in unique])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 2.5, maxIters=3000, confidence=.999)
    if H is None or mask is None or mask.sum() < 16:
        return None
    valid = mask.ravel().astype(bool)
    err = np.linalg.norm(cv2.perspectiveTransform(pa[:, None], H)[:, 0] - pb, axis=1)[valid]
    def coverage(points, shape):
        return float(cv2.contourArea(cv2.convexHull(points)) / (shape[0] * shape[1]))
    return dict(ratio_matches=len(unique), inliers=int(valid.sum()),
                inlier_fraction=float(valid.mean()), median_error_px=float(np.median(err)),
                coverage_a=coverage(pa[valid], a['shape']),
                coverage_b=coverage(pb[valid], b['shape']), H=H.tolist(),
                points_a=pa[valid].tolist(), points_b=pb[valid].tolist())


def scan(args):
    inv = json.loads((args.output / 'inventory.json').read_text())
    rows = inv['records']
    args.cache.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cache = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (r, ft) in enumerate(zip(rows, pool.map(lambda r: features(r, args.cache), rows))):
            cache[r['id']] = ft
            if i % 100 == 0:
                print('Features', i, '/', len(rows), round(time.time()-started, 1), 'seconds', flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['board_prefix'])].append(row)
    # Filename is used only to choose one representative of each exported view,
    # never to decide whether two views are alike.
    reps = [max(g, key=lambda r: len(cache[r['id']]['points'])) for g in groups.values()]
    pairs = {tuple(sorted([a['id'], b['id']])) for a, b in itertools.combinations(reps, 2)
             if a['split'] != b['split']}
    exhaustive_pairs = len(pairs)
    # Every individual variant also gets rotation/reflection-invariant retrieval.
    for i, a in enumerate(rows):
        others = []
        for b in rows:
            if a['split'] == b['split']:
                continue
            distance = min((a['phashes'][0] ^ h).bit_count() for h in b['phashes'])
            others.append((distance, b['id']))
        for _, other in sorted(others)[:5]:
            pairs.add(tuple(sorted([a['id'], other])))
    grouped_pairs = defaultdict(list)
    for a, b in sorted(pairs):
        grouped_pairs[b].append(a)
    rows_by_id = {r['id']: r for r in rows}
    candidates = []
    done = 0
    for b, aas in grouped_pairs.items():
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        matcher.add([cache[b]['descriptors']]); matcher.train()
        for a in aas:
            r = register(cache[a], cache[b], matcher)
            if r:
                r.update(a=a, b=b,
                         both_active_in_r4=rows_by_id[a]['active_in_r4'] and rows_by_id[b]['active_in_r4'])
                r['strong_geometry_candidate'] = (r['inliers'] >= 40 and
                    r['inlier_fraction'] >= .55 and min(r['coverage_a'], r['coverage_b']) >= .08)
                candidates.append(r)
            done += 1
            if done % 1000 == 0:
                print('Registered', done, '/', len(pairs), 'candidates', len(candidates),
                      'elapsed', round(time.time()-started, 1), flush=True)
        # Checkpoint progress in an audit-only output directory.
        save(args.output / 'registration_progress.json', dict(done=done, total=len(pairs),
                                                             candidates=len(candidates)))
    candidates.sort(key=lambda r: -r['inliers'])
    report = dict(method='SIFT ratio .70, unique target matches, homography RANSAC 2.5px; visual adjudication required',
                  feature_size=1100, max_features=1800, representative_views=len(reps),
                  exhaustive_cross_split_representative_pairs=exhaustive_pairs,
                  all_variant_retrieval_extra_pairs=len(pairs)-exhaustive_pairs,
                  total_registered_pairs=done, elapsed_seconds=time.time()-started,
                  strong_candidates=sum(r['strong_geometry_candidate'] for r in candidates),
                  candidates=candidates)
    save(args.output / 'registration_candidates.json', report)
    print('SCAN COMPLETE', {k: v for k, v in report.items() if k != 'candidates'}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['inventory', 'scan'])
    p.add_argument('--source', type=Path, default=Path('data_review/native1280_v2_20261106/source'))
    p.add_argument('--revised', type=Path, default=Path('datasets/pcb-native-reviewed-r4-final'))
    p.add_argument('--previous', type=Path, default=Path('data_review/native1280_v2_20261106/audit/inventory.json'))
    p.add_argument('--champion-manifest', type=Path, default=Path('results/champion_audit_v1_20261030/manifest.json'))
    p.add_argument('--output', type=Path, default=Path('data_review/cleaned1280_deep_audit_20261115'))
    p.add_argument('--cache', type=Path, default=Path('/tmp/pcb-cleaned-leakage-features-20261115'))
    args = p.parse_args()
    inventory(args) if args.stage == 'inventory' else scan(args)


if __name__ == '__main__':
    main()
