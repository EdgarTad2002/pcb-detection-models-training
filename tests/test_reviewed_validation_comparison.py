"""Guard against comparing different labels, settings or checkpoint files."""
import copy
import hashlib
from pathlib import Path
import pytest
from tools.evaluate_reviewed_validation import verify_records, summarize, SETTINGS, runtime_observations
from tools.controlled_experiments import split_signature


def test_callback_uses_actual_precision_without_validator_model_attribute():
    from types import SimpleNamespace
    validator=SimpleNamespace(seen=78,args=SimpleNamespace(quantize=None,rect=True,split='val'),
                              metrics=SimpleNamespace(nt_per_class=[1,2,3,4]))
    assert runtime_observations(validator)['fp16'] is False
    assert runtime_observations(validator)['target_counts']==[1,2,3,4]
    validator.args.quantize=16
    assert runtime_observations(validator)['fp16'] is True


def test_same_size_label_edit_is_rejected(tmp_path):
    image=tmp_path/'board.jpg';image.write_bytes(b'unchanged pixels')
    label=tmp_path/'board.txt';label.write_text('0 0.5 0.5 0.1 0.1\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    records=[dict(name='board.jpg',image=str(image),label=str(label),
                  image_sha256=sha(image),label_sha256=sha(label))]
    fingerprint=split_signature(records);verify_records(records,fingerprint)
    label.write_text('2 0.5 0.5 0.1 0.1\n')
    with pytest.raises(ValueError,match='Validation labels changed'):
        verify_records(records,fingerprint)


def test_no_mixed_protocol_scores_or_replaced_checkpoints():
    cases=[dict(id=a,arm=a,model='model',checkpoint_sha256=a) for a in ['old','retrained']]
    manifest=dict(cases=cases,settings=SETTINGS,validation_fingerprint='labels',
                  validation_image_count=78,disclosure='pilot')
    results=[dict(**c,settings=copy.deepcopy(SETTINGS),validation_fingerprint='labels',
                  mAP50=v,mAP50_95=v/2) for c,v in zip(cases,[.4,.5])]
    assert summarize(manifest,results)['pairs'][0]['mAP50_change_percentage_points']==pytest.approx(10)
    changed=copy.deepcopy(results);changed[0]['settings']['split']='test'
    with pytest.raises(ValueError,match='Mixed'):summarize(manifest,changed)
    changed=copy.deepcopy(results);changed[0]['checkpoint_sha256']='substituted'
    with pytest.raises(ValueError,match='Mixed'):summarize(manifest,changed)
