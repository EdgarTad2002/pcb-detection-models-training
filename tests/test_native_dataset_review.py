import json
from pathlib import Path
from PIL import Image
import pytest
from tools.audit_native_dataset import apply, sha, read_boxes


def fixture(tmp_path):
    im=tmp_path/'source.png';Image.new('RGB',(32,32),'green').save(im)
    label=tmp_path/'source.txt';label.write_text('0 0.5 0.5 0.25 0.25\n')
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps(dict(records=[dict(id='train/source.png',split='train',image=str(im),label=str(label),image_sha256=sha(im),label_sha256=sha(label))])))
    patches=tmp_path/'patches.json';patches.write_text(json.dumps([dict(id='train/source.png',action='replace',line=1,expected=label.read_text().strip(),replacement='2 0.5 0.5 0.25 0.25',reason='Verified electrolytic',evidence='review crop')]))
    return inventory,patches,label


def test_correction_copies_labels_without_modifying_source(tmp_path):
    inventory,patches,label=fixture(tmp_path);before=label.read_bytes();out=tmp_path/'candidate'
    apply(inventory,patches,out)
    assert label.read_bytes()==before
    assert (out/'train/labels/source.txt').read_text().startswith('2 ')
    assert (out/'train/images/source.png').is_symlink()
    with pytest.raises(FileExistsError):apply(inventory,patches,out)


def test_changed_source_refused_before_output_is_created(tmp_path):
    inventory,patches,label=fixture(tmp_path);label.write_text('0 0.4 0.4 0.1 0.1\n');out=tmp_path/'candidate'
    with pytest.raises(ValueError,match='Source changed'):apply(inventory,patches,out)
    assert not out.exists()


def test_quarantined_image_cannot_enter_training(tmp_path):
    inventory,patches,label=fixture(tmp_path);excluded=tmp_path/'excluded.json';excluded.write_text(json.dumps({'train/source.png':{'reason':'same board in test'}}));out=tmp_path/'candidate'
    apply(inventory,patches,out,excluded)
    assert not (out/'train/images/source.png').exists()
    assert (out/'quarantine/train/images/source.png').is_file()
    assert (out/'quarantine/train/labels/source.txt').read_text().startswith('2 ')


def test_ambiguous_or_malformed_rows_are_flagged_instead_of_silently_parsed(tmp_path):
    label=tmp_path/'bad.txt';label.write_text('1.5 0.5 0.5 0.2 0.2\n0 0.5 0.5 -0.2 0.2\n0 nan 0.5 0.2 0.2\n')
    boxes,issues=read_boxes(label)
    assert boxes==[]
    assert len(issues)==3
