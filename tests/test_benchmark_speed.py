"""Regression coverage for benchmark provenance, timing, selection and pipeline scope."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from tools import benchmark_speed as bench


class FakeModel:
    def __init__(self, boxes=None, half=False):
        self.calls = []
        self.predictor = SimpleNamespace(model=SimpleNamespace(fp16=half, device='cpu', end2end=False))
        self.boxes = boxes or []

    def predict(self, image, **kwargs):
        self.calls.append((image.shape, kwargs))
        return [SimpleNamespace(speed={'inference': 2.}, boxes=self.boxes)]


class BenchmarkTest(unittest.TestCase):
    def args(self, *extra):
        return bench.parse_args(['--weights', 'runs/a/weights/best.pt', '--device', 'cpu',
                                 '--imgsz', '64', '--warmup', '2', '--num-iters', '7', *extra])

    def test_duplicate_best_filenames_have_unique_stable_ids(self):
        root = Path('/tmp/project')
        paths = [root / f'runs/{name}/pcb-filtered/weights/best.pt' for name in ('one', 'two')]
        labels = [bench.checkpoint_id(p, root) for p in paths]
        self.assertNotEqual(*labels)
        self.assertEqual(labels[0], bench.checkpoint_id(paths[0], root))

    def test_loads_mixed_case_jpeg_and_rejects_missing_or_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (64, 64), (255, 0, 0)).save(root / 'one.JPEG')
            args = self.args('--images-dir', str(root))
            images, metadata = bench.load_inputs(args)
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0][0, 0].tolist(), [0, 0, 254])
            self.assertEqual(metadata['kind'], 'real_decoded_boards')
            self.assertEqual(len(metadata['source_images'][0]['sha256']), 64)
            args.images_dir = root / 'missing'
            with self.assertRaises(FileNotFoundError):
                bench.load_inputs(args)
            args.images_dir = root / 'empty'
            args.images_dir.mkdir()
            with self.assertRaisesRegex(ValueError, 'No real'):
                bench.load_inputs(args)

    def test_exact_timed_iterations_repeat_cached_images_and_forward_options(self):
        args = self.args()
        model = FakeModel()
        predictor = bench.Predictor(model, args, False)
        images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        result = bench.measure([predictor], images, args, torch.device('cpu'))
        self.assertEqual(len(model.calls), 1 + args.warmup + args.num_iters)
        self.assertEqual(result['wall_prediction']['count'], 7)
        self.assertEqual(result['detector_calls_per_board'], [1] * 7)
        self.assertEqual(result['backend_reported_forward_sum']['samples_ms'], [2.] * 7)
        self.assertAlmostEqual(result['fps_from_mean_latency'], 1000 / result['wall_prediction']['mean_ms'])
        for _, options in model.calls:
            self.assertEqual(options['iou'], args.bench_iou)
            self.assertEqual(options['conf'], args.bench_conf)
            self.assertFalse(options['rect'])
            self.assertEqual(options['batch'], 1)
            self.assertIn(options.get('quantize', options.get('half')), (None, False))

    def test_gpu_measurement_uses_selected_device_and_reports_distinct_timings(self):
        args = self.args('--precision', 'fp16')
        args.device = '1'
        model = FakeModel(half=True)
        model.predictor.model.device = 'cuda:1'
        predictor = bench.Predictor(model, args, True)
        event = SimpleNamespace(record=lambda stream: None, elapsed_time=lambda other: 3.)
        device = torch.device('cuda:1')
        with patch.object(bench.torch.cuda, 'synchronize') as sync, \
             patch.object(bench.torch.cuda, 'Event', return_value=event), \
             patch.object(bench.torch.cuda, 'current_stream', return_value='stream1') as stream, \
             patch.object(bench, 'gpu_isolation', return_value={'status': 'checked'}):
            result = bench.measure([predictor], [np.zeros((64, 64, 3))], args, device)
        self.assertTrue(all(call.args == (device,) for call in sync.call_args_list))
        stream.assert_called_with(device)
        self.assertEqual(result['cuda_pipeline_interval']['samples_ms'], [3.] * 7)
        self.assertEqual(result['backend_reported_forward_sum']['samples_ms'], [2.] * 7)
        self.assertTrue(all(c[1].get('quantize', c[1].get('half')) in (16, True) for c in model.calls))

    def test_precision_mismatch_fails_instead_of_mislabeling_fp16(self):
        args = self.args()
        predictor = bench.Predictor(FakeModel(half=False), args, True)
        with self.assertRaisesRegex(RuntimeError, 'precision'):
            bench.measure([predictor], [np.zeros((64, 64, 3))], args, torch.device('cpu'))

    def test_backend_device_mismatch_is_rejected(self):
        args = self.args()
        model = FakeModel()
        model.predictor.model.device = 'cuda:1'
        with self.assertRaisesRegex(RuntimeError, 'Backend device'):
            bench.Predictor(model, args, False).check_precision()

    def test_gpu_sharing_is_rejected_and_explicit_override_is_recorded(self):
        query = SimpleNamespace(stdout=f'GPU-one, {os.getpid()}, own, 10\nGPU-one, 999999, training, 5000\nGPU-two, 88888, othergpu, 100\n')
        with patch.object(bench.torch.cuda, 'get_device_properties', return_value=SimpleNamespace(uuid='GPU-one')), \
             patch.object(bench.subprocess, 'run', side_effect=lambda cmd, **kw: SimpleNamespace(stdout='GPU-one, 590.00\n') if '--query-gpu=uuid,driver_version' in cmd else query):
            with self.assertRaisesRegex(RuntimeError, 'isolation'):
                bench.gpu_isolation(torch.device('cuda:1'))
            info = bench.gpu_isolation(torch.device('cuda:1'), True)
            self.assertEqual(len(info['other_compute_processes']), 1)
            self.assertTrue(info['sharing_override'])

    def test_bare_torch_uuid_matches_nvidia_prefix_without_hiding_other_processes(self):
        def query(command, **kwargs):
            if '--query-gpu=uuid,driver_version' in command:
                return SimpleNamespace(stdout='GPU-a688a91d-0267-7ac9-53a7-8db8dde8b836, 590.00\n')
            return SimpleNamespace(stdout=f'GPU-a688a91d-0267-7ac9-53a7-8db8dde8b836, {os.getpid()}, own, 10\n')
        with patch.object(bench.torch.cuda, 'get_device_properties', return_value=SimpleNamespace(uuid='a688a91d-0267-7ac9-53a7-8db8dde8b836')), \
             patch.object(bench.subprocess, 'run', side_effect=query):
            info = bench.gpu_isolation(torch.device('cuda:0'))
            self.assertEqual(info['status'], 'no_other_compute_processes_observed')
            self.assertTrue(info['uuid'].startswith('GPU-'))

    def test_crop_pipeline_counts_all_tiles_and_overrides_hardcoded_precision(self):
        from tools.eval_dual_branch_pipeline import get_slice_boxes
        args = self.args('--pipeline', 'crop', '--crop-weights', 'crop.pt', '--slice-size', '32', '--precision', 'fp16')
        models = [FakeModel(half=True), FakeModel(half=True)]
        predictors = [bench.Predictor(m, args, True) for m in models]
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        result = bench.measure(predictors, [image], args, torch.device('cpu'))
        expected = 1 + len(get_slice_boxes(80, 64, 32, 32, .2))
        self.assertEqual(result['detector_calls_per_board'], [expected] * 7)
        self.assertTrue(all(kwargs.get('quantize', kwargs.get('half')) in (16, True) for model in models for _, kwargs in model.calls))

    def test_ensemble_fusion_is_inside_timed_pipeline(self):
        args = bench.parse_args(['--weights', 'a.pt', 'b.pt', '--pipeline', 'ensemble', '--device', 'cpu',
                                 '--warmup', '0', '--num-iters', '3'])
        box = SimpleNamespace(xyxy=torch.tensor([[0., 0., 10., 10.]]), conf=torch.tensor([.9]), cls=torch.tensor([0.]))
        models = [FakeModel(), FakeModel()]
        for m in models:
            m.boxes = box
        predictors = [bench.Predictor(m, args, False) for m in models]
        import torchvision
        original = torchvision.ops.batched_nms
        with patch.object(torchvision.ops, 'batched_nms', wraps=original) as fusion:
            result = bench.measure(predictors, [np.zeros((64, 64, 3))], args, torch.device('cpu'))
        self.assertEqual(fusion.call_count, 4)  # one backend initialization, three timed boards
        self.assertEqual(result['detector_calls_per_board'], [2] * 3)

    def test_accuracy_class_indices_are_mapped_and_split_is_saved(self):
        args = self.args('--re-evaluate-map')
        names = ['Capacitor', 'Connector', 'Electrolytic Capacitor', 'IC']
        metrics = SimpleNamespace(names=dict(enumerate(names)), box=SimpleNamespace(
            map50=.3, map=.2, ap_class_index=np.array([0, 3]), ap50=np.array([.1, .5]), ap=np.array([.05, .4])))
        model = SimpleNamespace(names=dict(enumerate(names)), val=lambda **kw: metrics)
        data = {'yaml': '/data.yaml', 'splits': {'val': []}}
        with patch('tools.controlled_experiments.read_dataset', return_value=data):
            result = bench.evaluate_accuracy(model, args, Path('/tmp/accuracy'))
        self.assertEqual(result['split'], 'val')
        self.assertEqual(result['per_class_ap50'], {'Capacitor': .1, 'IC': .5})
        self.assertIn('split_fingerprint', result)

    def test_wbf_coordinates_and_member_weights_are_preserved(self):
        args = bench.parse_args(['--weights', 'a.pt', 'b.pt', '--pipeline', 'ensemble', '--device', 'cpu',
                                 '--fusion', 'wbf', '--ensemble-weights', '1', '2'])
        boxes = SimpleNamespace(xyxy=torch.tensor([[0., 0., 10., 10.]]), conf=torch.tensor([.9]), cls=torch.tensor([0.]))
        models = [FakeModel(), FakeModel()]
        for model in models:
            model.boxes = boxes
        from unittest.mock import Mock
        fusion = Mock(return_value=(np.array([[.1, .2, .3, .4]]), np.array([.9]), np.array([0.])))
        with patch.dict('sys.modules', {'ensemble_boxes': SimpleNamespace(weighted_boxes_fusion=fusion)}):
            output = bench.ensemble_inference([bench.Predictor(m, args, False) for m in models],
                                              np.zeros((50, 100, 3)), args)
        self.assertTrue(np.allclose(output[0], [[10., 10., 30., 20.]]))
        self.assertEqual(fusion.call_args.kwargs['weights'], [1., 2.])

    def test_finished_checkpoint_requires_final_curve_and_matching_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'weights/best.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'checkpoint')
            result = root / 'result.json'
            result.write_text(json.dumps({'weights': str(checkpoint), 'eval_split': 'val'}))
            (root / 'results.csv').write_text('epoch\n1\n')
            with self.assertRaisesRegex(ValueError, 'planned epochs'):
                bench.assert_completed(checkpoint, result, 2)
            (root / 'results.csv').write_text('epoch\n1\n2\n')
            bench.assert_completed(checkpoint, result, 2)
            result.write_text(json.dumps({'weights': str(checkpoint), 'eval_split': 'test'}))
            with self.assertRaisesRegex(ValueError, 'Completion result'):
                bench.assert_completed(checkpoint, result, 2)

    def test_default_bundle_selects_new_and_reused_controls_without_legacy_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'controlled_experiments'
            directory.mkdir()
            plans = []
            for name in ('new', 'old'):
                output = directory / name
                output.mkdir()
                plan = {'root': str(root), 'output': str(output),
                        'config': {'ultralytics_version': 'test-version'},
                        'jobs': [{'key': name, 'epochs': 100}]}
                path = output / 'plan.json'
                path.write_text(json.dumps(plan))
                plans.append(path)
                weights = root / 'runs' / name / 'pcb-filtered/weights/best.pt'
                weights.parent.mkdir(parents=True)
                weights.write_bytes(b'checkpoint')
            bundle = {'new_plans': [str(plans[0])], 'control_plan': str(plans[1]),
                      'reused_controls': [{'weights': str(root / 'runs/old/pcb-filtered/weights/best.pt'),
                                           'validation_result': str(directory / 'old/result.json')}]}
            (directory / 'clean_main_noextraaug_v1_bundle.json').write_text(json.dumps(bundle))
            args = bench.parse_args(['--project-root', str(root), '--device', 'cpu'])
            with patch('tools.controlled_experiments.verify_plan') as verify, \
                 patch.object(bench, 'assert_completed') as completed, \
                 patch('importlib.metadata.version', return_value='test-version'):
                selected = bench.select_checkpoints(args)
            self.assertEqual(len(selected), 2)
            self.assertEqual(verify.call_count, 2)
            self.assertEqual(completed.call_count, 2)
            self.assertTrue(any('/old/' in str(p) for p in selected))

    def test_removed_update_mode_is_rejected_and_accuracy_defaults_to_validation(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('--update-results')
        args = self.args()
        self.assertFalse(args.re_evaluate_map)
        self.assertEqual(args.eval_split, 'val')
        self.assertEqual(args.input_split, 'val')

    def test_synthetic_requires_explicit_request(self):
        args = self.args('--synthetic')
        images, metadata = bench.load_inputs(args)
        self.assertEqual(images[0].shape, (64, 64, 3))
        self.assertEqual(metadata['kind'], 'synthetic_diagnostic')

    def test_crop_rejects_wrong_class_mapping(self):
        models = [SimpleNamespace(names={0: 'person', 1: 'car'})] * 2
        with self.assertRaisesRegex(ValueError, 'four-class'):
            bench.validate_pipeline_classes(models, 'crop')
        with self.assertRaisesRegex(ValueError, 'different class'):
            bench.validate_pipeline_classes([models[0], SimpleNamespace(names={0: 'other'})], 'ensemble')

    def test_failures_are_saved_with_nonzero_exit_without_touching_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accuracy = root / 'original_accuracy.json'
            accuracy.write_text('{"mAP50": 0.65, "eval_split": "test"}')
            weights = []
            for name in ('one', 'two'):
                path = root / name / 'best.pt'
                path.parent.mkdir()
                path.write_bytes(b'bad checkpoint')
                weights.append(str(path))
            (root / 'boards').mkdir()
            Image.new('RGB', (64, 64)).save(root / 'boards/image.png')
            argv = ['--weights', *weights, '--device', 'cpu', '--images-dir', str(root / 'boards'),
                    '--output-dir', str(root / 'speed')]
            with patch.object(bench, 'register_modules', return_value={}), \
                 patch('ultralytics.YOLO', side_effect=RuntimeError('broken checkpoint')), redirect_stdout(io.StringIO()):
                self.assertEqual(bench.main(argv), 1)
                with self.assertRaises(FileExistsError):
                    bench.main(argv)
            result = json.loads((root / 'speed/benchmark.json').read_text())
            self.assertEqual(len(result['failures']), 2)
            self.assertEqual(len({r['id'] for r in result['records']}), 2)
            self.assertEqual(json.loads(accuracy.read_text()), {'mAP50': .65, 'eval_split': 'test'})


if __name__ == '__main__':
    unittest.main()
