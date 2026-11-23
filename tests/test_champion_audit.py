import json
from pathlib import Path
import numpy as np
import pytest

from tools.evaluate_champion_audit import ROOT, fuse, score


def test_exact_files_and_prespecified_ensembles():
    cfg = json.loads((ROOT / 'configs/champion_audit.json').read_text())
    assert len({m['sha256'] for m in cfg['models']}) == 4
    assert [len(e['members']) for e in cfg['ensembles']] == [2, 4]
    assert cfg['settings']['conf'] == .001


def test_ap_and_operating_recall_are_distinct(tmp_path):
    label = tmp_path / 'board.txt'
    label.write_text('0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.2 0.2\n')
    predictions = [[[30, 30, 70, 70, .9, 0], [10, 10, 30, 30, .1, 1]]]
    r = score(predictions, [{'label': str(label)}], [[100, 100]], {'recall_conf': .25})
    assert r['mAP50'] > .99
    assert r['mAP50_95'] > .99
    assert r['per_class']['Capacitor']['recall'] == 1
    assert r['per_class']['Connector']['recall'] == 0
    assert r['per_class']['Connector']['ap50'] > .99


def test_empty_predictions_and_wrong_classes(tmp_path):
    label = tmp_path / 'board.txt'
    label.write_text('0 0.5 0.5 0.4 0.4\n')
    settings = {'recall_conf': .25}
    r = score([[]], [{'label': str(label)}], [[100, 100]], settings)
    assert r['mAP50'] == 0
    assert r['per_class']['Capacitor']['fn'] == 1
    wrong = score([[[30, 30, 70, 70, .9, 1]]], [{'label': str(label)}], [[100, 100]], settings)
    assert wrong['mAP50'] == 0
    assert wrong['per_class']['Connector']['fp'] == 1


def test_wbf_pixel_conversion_empty_member_and_global_cap():
    settings = {'merge_iou': .5, 'conf': .001, 'max_det': 1}
    p = [[20, 10, 60, 30, .9, 0], [100, 50, 160, 80, .8, 1]]
    result = fuse([p, []], [100, 200], [1, 1], settings)
    assert len(result) == 1
    assert np.allclose(result[0][:4], p[0][:4])
    assert result[0][4] == pytest.approx(.45)
    assert fuse([[], []], [100, 200], [1, 1], settings) == []
