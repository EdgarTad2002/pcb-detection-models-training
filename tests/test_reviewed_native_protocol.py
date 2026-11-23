"""Guard the reviewed dataset, fresh initialization, and comparable crop budget."""
import json
from pathlib import Path

from tools.controlled_experiments import command_for, make_jobs


def test_reviewed_round_uses_five_fresh_jobs_and_only_validation():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/reviewed_native_r3_experiments.json').read_text())
    jobs = make_jobs(config, ['main', 'crop'], counts={'clean':492, 'mixed':984})
    assert len(jobs) == 5
    assert {j['seed'] for j in jobs} == {0}
    assert {j['epochs'] for j in jobs if j['dataset'] == 'clean'} == {100}
    crop = next(j for j in jobs if j['dataset'] == 'mixed')
    assert crop['epochs'] == 50
    assert abs(crop['budget']['relative_difference']) < .02
    assert config['datasets']['clean'].endswith('pcb-native-res-reviewed-v2-candidate-20261106-r3/data.yaml')
    for job in jobs:
        model = config['models'][job['model']]
        assert model['weights'] in {'yolo26s.pt', 'yolo12s.pt'}
        assert 'pretrained_weights' not in model
        cmd = command_for(config, job, root, root/'output', Path('train.yaml'), Path('eval.yaml'))
        assert cmd[cmd.index('--eval-split')+1] == 'val'
        assert '--skip-train' not in cmd
    full = config['models']['yolo26s_agwd_sdice']
    assert config['models']['yolo26s_crop_specialist'] == full
    assert (full['lambda_dice'], full['smooth_dice'], full['stage_weights']) == (.5,.01,[1.,1.,1.])
    assert (full['agwd_kappa'],full['agwd_tau']) == (12.8,576.)
    assert 'incomplete' in config['dataset_review_status']
