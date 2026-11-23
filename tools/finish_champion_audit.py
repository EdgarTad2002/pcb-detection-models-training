#!/usr/bin/env python3
"""Retrieve the authorized cluster audit and rebuild its reports when complete."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training'
RUNTIME = Path('/home/edgar/.cache/codex-runtimes/codex-primary-runtime/dependencies')
SKILLS = Path('/home/edgar/.codex/plugins/cache/openai-primary-runtime')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True, type=int)
    p.add_argument('--directory', default='champion_audit_v1_20261030')
    p.add_argument('--reports', type=Path, help='Optional new output directory for a fresh report version')
    p.add_argument('--build-only', action='store_true', help='Use previously retrieved summary.json without SSH')
    a = p.parse_args()
    if Path(a.directory).name != a.directory:
        raise ValueError('Directory must be a simple name')
    local = ROOT / 'results' / a.directory
    local.mkdir(parents=True, exist_ok=True)
    status = local / 'report_status.json'
    previous = json.loads(status.read_text()) if status.is_file() else {}
    reports = (a.reports or Path(previous.get('reports', ROOT / 'paper/reports' / (a.directory + '_verified')))).resolve()
    if previous.get('state') == 'completed' and not a.reports:
        proof = json.loads((reports / 'export_provenance.json').read_text())
        summary_hash = hashlib.sha256((local / 'summary.json').read_bytes()).hexdigest()
        if summary_hash != proof['source_summary_sha256']:
            raise ValueError('Saved summary differs from the completed report source')
        for name, expected in proof['output_sha256'].items():
            if hashlib.sha256((reports / name).read_bytes()).hexdigest() != expected:
                raise ValueError('A completed report changed; choose a new --reports directory to rebuild')
        print(json.dumps(previous), flush=True)
        return

    def record(state, **extra):
        value = dict(state=state, job=a.job, reports=str(reports), **extra)
        status.write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(value), flush=True)

    remote = f'{REMOTE}/results/{a.directory}'
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'cluster.ysu.am']
    try:
        if a.build_only and not (local / 'summary.json').is_file():
            raise FileNotFoundError('No local summary.json is available')
        if not a.build_only:
            record('waiting_for_cluster_results')
        while not a.build_only:
            check = subprocess.run([*ssh, f'test -f {remote}/summary.json'], capture_output=True)
            if check.returncode == 0:
                break
            state = subprocess.run([*ssh, f'sacct -j {a.job} -n -X --format=State --parsable2'], capture_output=True, text=True)
            if state.returncode == 0:
                text = state.stdout.strip()
                if any(s in text for s in ['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'COMPLETED']):
                    raise RuntimeError(f'Job ended without a complete summary: {text}')
            time.sleep(45)
        if not a.build_only:
            record('retrieving_results')
            subprocess.run(['scp', '-r', f'cluster.ysu.am:{remote}/.', str(local)], check=True)
        record('building_documents')
        node = str(RUNTIME / 'node/bin/node')
        # Each marker runs once immediately before authoring the corresponding artifact.
        for kind, folder in [('xlsx', 'spreadsheets/26.1020.11618/skills/spreadsheets'),
                             ('pptx', 'presentations/26.1020.11618/skills/presentations')]:
            subprocess.run([node, 'container_tools/mark_artifact_operation_started.mjs',
                            '--operation-kind', 'create', '--expected-output-count', '1', '--output-format', kind],
                           cwd=SKILLS / folder, check=True)
        with tempfile.TemporaryDirectory(prefix='champion-audit-reports-') as tmp:
            directory = Path(tmp)
            (directory / 'node_modules').symlink_to(RUNTIME / 'node/node_modules')
            builder = directory / 'build.mjs'
            shutil.copyfile(ROOT / 'tools/build_champion_audit_documents.mjs', builder)
            subprocess.run([node, str(builder), str(local / 'summary.json'), str(reports)], check=True)
        record('generated_pending_visual_review', workbook=str(reports / 'PCB_Detection_Master_Benchmark_Verified.xlsx'),
               presentation=str(reports / 'Professor_Meeting_Presentation_Verified.pptx'))
    except Exception as exc:
        record('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
