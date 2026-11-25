"""Run the README example benchmark and export traceable paired measurements.

Run from the repository root after installing .[metrics]. All variants run
sequentially on the same visible GPU. Dataset-scale evaluation is separate.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {'dense': ('dense', 'r20'), 'r40': ('sdit', 'r40'), 'r20': ('sdit', 'r20')}


def read(path):
    return json.loads(path.read_text())


def collect(output):
    result = {'model': 'Alpha-VLLM/Lumina-Next-SFT-diffusers',
              'scope': 'Two fixed example prompts, 30 steps, 1024x1024, CFG 4',
              'timing': 'Same GPU; one excluded warm-up and two measured repeats per prompt; synchronized pipeline through PIL conversion',
              'quality': 'Mean paired PSNR, SSIM and LPIPS-Alex on uint8 RGB outputs against dense generation',
              'memory': 'Maximum torch.cuda.max_memory_allocated across measured runs, in GiB',
              'variants': {}}
    engine_hashes = None
    for variant in VARIANTS:
        timed, generated = output/(variant+'-timing'), output/(variant+'-images')
        tm, gm = read(timed/'run.json'), read(generated/'run.json')
        rows, images = read(timed/'results.json'), read(generated/'results.json')
        assert tm['config'] == gm['config'] and tm['prompts'] == gm['prompts']
        assert len(rows) == 2*len(images), 'Expected two timed repeats per generated image'
        hashes = {k:v for k,v in tm['source_sha256'].items()
                  if k.startswith('ras/') or k in ('sdit/config.py','sdit/engine.py','sdit/policy.py')}
        if engine_hashes is None:
            engine_hashes = hashes
            result.update(environment=tm['environment'], prompts=tm['prompts'], inference_source_sha256=hashes)
        assert hashes == engine_hashes, 'Inference implementation changed between variants'
        assert result['environment']['gpu'] == tm['environment']['gpu']
        for image in images:
            matches = [r for r in rows if r['id'] == image['id']]
            assert len(matches) == 2
            assert all(r['pixel_sha256'] == image['pixel_sha256'] for r in matches), 'Timing/quality outputs differ'
        metrics = read(output/(variant+'-quality.json')) if variant != 'dense' else None
        result['variants'][variant] = {
            'config': tm['config'], 'mean_seconds': statistics.mean(r['wall_seconds'] for r in rows),
            'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in rows)/2**30,
            'timed_runs': [{k:r[k] for k in ('id','repeat','wall_seconds','pixel_sha256')} for r in rows],
            'quality': metrics['summary'] if metrics else None,
            'quality_pairs': metrics['pairs'] if metrics else None}
    dense_time = result['variants']['dense']['mean_seconds']
    for data in result['variants'].values():
        data['speedup'] = dense_time/data['mean_seconds']
    (output/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{f:v[f] for f in ('mean_seconds','speedup','peak_allocated_gib','quality')}
                      for k,v in result['variants'].items()},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/readme'))
    p.add_argument('--model-path')
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--collect-only',action='store_true',help='Evaluate and summarize existing benchmark/generate outputs')
    args=p.parse_args()
    output=args.output.resolve()
    if not args.collect_only and output.exists() and any(output.iterdir()):
        p.error('Choose a new or empty output directory')
    if not args.collect_only:
        output.mkdir(parents=True,exist_ok=True)
        for variant,(method,ratio) in VARIANTS.items():
            common=['--config',f'configs/lumina_sdit_{ratio}.json','--method',method,
                    '--prompts','examples/prompts.jsonl']
            if args.model_path: common += ['--model-path',args.model_path]
            if args.local_files_only: common += ['--local-files-only']
            for command,suffix,extra in [('benchmark','timing',['--warmup-runs','1','--repeats','2']),('generate','images',[])]:
                subprocess.run([sys.executable,'-m','sdit.cli',command,*common,*extra,
                                '--output',str(output/(variant+'-'+suffix))],cwd=ROOT,check=True)
    for variant in ('r40','r20'):
        subprocess.run([sys.executable,'-m','sdit.cli','evaluate','--generated',str(output/(variant+'-images')),
                        '--dense',str(output/'dense-images'),'--lpips','--cpu',
                        '--output',str(output/(variant+'-quality.json'))],cwd=ROOT,check=True)
    collect(output)


if __name__ == '__main__':
    main()
