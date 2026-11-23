"""Tests for frozen test comparisons, complete seed coverage and safe result reuse."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import evaluate_clean_main_test as evaluation
from tools.controlled_experiments import ROOT, NAMES


class TestEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/clean_main_experiments.json').read_text())

    def manifest_and_results(self):
        settings = {'split': 'test', 'imgsz': 1280, 'conf': .001, 'iou': .5, 'max_det': 1000}
        cases, results = [], []
        for model in self.config['models']:
            for seed in (0, 1, 2):
                case = {'model': model, 'seed': seed, 'checkpoint_sha256': f'{model}_{seed}_hash'}
                cases.append(case)
                results.append({**case, 'eval_split': 'test', 'test_fingerprint': 'test_hash', 'settings': settings,
                    'mAP50': .5 + seed * .01, 'mAP50_95': .3 + seed * .01,
                    'per_class_ap50': {n: .2 + seed * .01 for n in NAMES},
                    'per_class_ap50_95': {n: .1 + seed * .01 for n in NAMES}})
        return {'cases': cases, 'settings': settings, 'dataset_fingerprints': {'test': 'test_hash'},
                'test_image_count': 44, 'disclosure': 'previously used test set'}, results

    def test_summary_requires_all_three_seeds_and_exports_capacitor_localization(self):
        manifest, results = self.manifest_and_results()
        summary = evaluation.summarize(manifest, results)
        self.assertEqual(summary['eval_split'], 'test')
        self.assertEqual(len(summary['summary']), 7)
        metric = summary['summary'][0]['metrics']['per_class_ap50_95/Capacitor']
        self.assertAlmostEqual(metric['mean'], 11.)
        self.assertAlmostEqual(metric['sample_sd'], 1.)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            evaluation.summarize(manifest, results[:-1])

    def test_validation_or_altered_checkpoint_results_cannot_be_reused_as_test(self):
        manifest, results = self.manifest_and_results()
        for field, value in [('eval_split', 'val'), ('checkpoint_sha256', 'changed'), ('test_fingerprint', 'changed')]:
            result = copy.deepcopy(results[0]); result[field] = value
            with self.assertRaisesRegex(ValueError, 'Existing result'):
                evaluation.check_result(result, manifest['cases'][0], manifest)
        result = copy.deepcopy(results[0]); result['settings']['conf'] = .25
        with self.assertRaisesRegex(ValueError, 'Existing result'):
            evaluation.check_result(result, manifest['cases'][0], manifest)

    def test_duplicate_seed_cannot_replace_missing_seed(self):
        manifest, results = self.manifest_and_results()
        results[-1] = results[-2]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            evaluation.summarize(manifest, results)

    def test_freezes_all_existing_validation_selected_checkpoints_including_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fingerprints = {'clean': {'train': 'a', 'val': 'b', 'test': 'c'}}
            plans = []
            for label in ('new', 'control'):
                out = root / label; out.mkdir()
                plans.append({'root': str(root), 'output': str(out), 'config': self.config,
                              'dataset_fingerprints': fingerprints, 'jobs': []})
            reused = []
            for model in self.config['models']:
                for seed in (0, 1, 2):
                    is_control = model in ('physot_yolo26s', 'yolov12s') and seed == 0
                    plan = plans[1 if is_control else 0]
                    key = f'{model}_s{seed}'
                    job = {'key': key, 'model': model, 'seed': seed, 'dataset': 'clean', 'epochs': 100}
                    plan['jobs'].append(job)
                    directory = root / 'runs' / key / 'pcb-filtered'
                    (directory / 'weights').mkdir(parents=True)
                    checkpoint = directory / 'weights/best.pt'
                    checkpoint.write_bytes(key.encode())
                    (directory / 'results.csv').write_text('epoch\n' + ''.join(f'{i}\n' for i in range(1, 101)))
                    target = Path(plan['output']) / 'validation' / (key + '.json')
                    target.parent.mkdir(exist_ok=True)
                    target.write_text(json.dumps({'seed': seed, 'epochs': 100, 'eval_split': 'val', 'weights': str(checkpoint)}))
                    if is_control:
                        reused.append({'job_key': key})
            paths = []
            for plan in plans:
                path = Path(plan['output']) / 'plan.json'; path.write_text(json.dumps(plan)); paths.append(path)
            bundle = {'new_plans': [str(paths[0])], 'control_plan': str(paths[1]), 'reused_controls': reused}
            with patch.object(evaluation, 'verify_plan'):
                cases, _, _ = evaluation.collect_cases(bundle, root)
            self.assertEqual(len(cases), 21)
            self.assertEqual(len({(c['model'], c['seed']) for c in cases}), 21)
            self.assertTrue(all(c['checkpoint'].endswith('/weights/best.pt') for c in cases))
            self.assertTrue(all(len(c['checkpoint_sha256']) == 64 for c in cases))
            bundle['reused_controls'].append(reused[0])
            with patch.object(evaluation, 'verify_plan'), self.assertRaisesRegex(ValueError, 'Duplicate'):
                evaluation.collect_cases(bundle, root)


if __name__ == '__main__':
    unittest.main()
