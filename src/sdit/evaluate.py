"""Paired dense-reference metrics with explicit FID reference sets."""
import argparse
import hashlib
import json
from pathlib import Path
import math

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from .cli import write_json


def load_run(directory):
    directory = Path(directory)
    meta = json.loads((directory / "run.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    if meta["command"] != "generate":
        raise ValueError("Quality evaluation expects generate outputs, not repeated benchmark images")
    rows = {r["id"]: r for r in results}
    prompts = {r["id"]: r for r in meta["prompts"]}
    if len(rows) != len(results) or set(rows) != set(prompts):
        raise ValueError("Incomplete or duplicate generation records")
    for row in results:
        filename = row["image"]
        if Path(filename).name != filename or not (directory / filename).is_file():
            raise ValueError("Invalid or missing recorded image")
        if "pixel_sha256" in row:
            with Image.open(directory / filename) as image:
                digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest != row["pixel_sha256"]:
                raise ValueError(f"Image changed since generation: {filename}")
    if {p.name for p in directory.glob('*.png')} != {r['image'] for r in results}:
        raise ValueError("Image directory contains unrecorded PNGs")
    return meta, rows, prompts


def validate_pair(generated, dense):
    gm, gr, gp = load_run(generated)
    dm, dr, dp = load_run(dense)
    if dm['config']['method'] != 'dense':
        raise ValueError("Dense reference run must use method=dense")
    if gp != dp:
        raise ValueError("Paired evaluation requires identical ids, prompts, negative prompts and seeds")
    for key in ('model', 'steps', 'height', 'width', 'guidance_scale'):
        if gm['config'][key] != dm['config'][key]:
            raise ValueError(f"Mismatched paired configuration: {key}")
    if gm['model'] != dm['model']:
        raise ValueError("Generated and dense runs must use the same recorded model identifier/path")
    return gm, gr, dr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generated', type=Path, required=True)
    p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lpips', action='store_true')
    p.add_argument('--fid', action='store_true', help='Compute FID_d against dense reference')
    p.add_argument('--real', type=Path, help='Explicit real-image reference directory for FID_r (requires --fid)')
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists; choose a new path')
    if a.real and not a.fid:
        p.error('--real requires --fid')
    try:
        meta, generated, dense = validate_pair(a.generated, a.dense)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    lpips_model = None
    if a.lpips:
        import torch
        import lpips
        device = 'cpu' if a.cpu else 'cuda'
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    rows = []
    for sample_id in sorted(generated):
        x = np.asarray(Image.open(a.generated / generated[sample_id]['image']).convert('RGB'), dtype=np.float32) / 255.
        y = np.asarray(Image.open(a.dense / dense[sample_id]['image']).convert('RGB'), dtype=np.float32) / 255.
        if x.shape != y.shape:
            raise ValueError(f'Shape mismatch for {sample_id}')
        mse = float(np.mean((x - y)**2, dtype=np.float64))
        row = dict(id=sample_id, psnr_db=None if mse == 0 else -10 * math.log10(mse),
                   identical=mse == 0, ssim=float(structural_similarity(x, y, data_range=1., channel_axis=2)))
        if lpips_model is not None:
            def tensor(z):
                return torch.from_numpy(z).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            with torch.inference_mode():
                row['lpips'] = float(lpips_model(tensor(x), tensor(y)).item())
        rows.append(row)
    # An exact pair has infinite PSNR: encode it explicitly rather than invalid JSON Infinity.
    summary = dict(count=len(rows), identical_pairs=sum(r['identical'] for r in rows),
                   psnr_db=None if any(r['identical'] for r in rows) else float(np.mean([r['psnr_db'] for r in rows])),
                   psnr_infinite=any(r['identical'] for r in rows), ssim=float(np.mean([r['ssim'] for r in rows])))
    if a.lpips:
        summary['lpips'] = float(np.mean([r['lpips'] for r in rows]))
    if a.fid:
        if len(rows) < 2:
            p.error('FID requires at least two samples; use the full evaluation set for meaningful estimates')
        import torch_fidelity
        def fid(reference):
            return float(torch_fidelity.calculate_metrics(input1=str(a.generated), input2=str(reference),
                cuda=not a.cpu, fid=True, isc=False, kid=False, verbose=False)['frechet_inception_distance'])
        summary['FID_d'] = fid(a.dense)
        if a.real:
            if not a.real.is_dir():
                p.error('--real must be an existing image directory')
            summary['FID_r'] = fid(a.real)
    result = dict(summary=summary, pairs=rows, generated=str(a.generated), dense_reference=str(a.dense),
                  real_reference=str(a.real) if a.real else None,
                  definitions={'FID_d': 'Generated set versus paired dense-generation set',
                               'FID_r': 'Generated set versus explicitly supplied real-image set'},
                  config=meta['config'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, result)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
