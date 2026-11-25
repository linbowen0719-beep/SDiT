import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from sdit.config import InferenceConfig
from sdit.cli import read_prompts
from sdit.evaluate import validate_pair


@pytest.mark.parametrize('steps,warmup', [(30,6),(15,3),(10,2),(7,2),(1,1)])
def test_warmup_uses_ceiling(steps, warmup):
    assert InferenceConfig(steps=steps).warmup == warmup


@pytest.mark.parametrize('values', [dict(rho=0),dict(rho=1.1),dict(rho=float('nan')),
    dict(steps=2.5),dict(height=1000),dict(anchors=-1),dict(warmup_steps=0),
    dict(model='hunyuan',backend='flash'),dict(analysis='false'),dict(guidance_scale=1)])
def test_invalid_configs_fail_before_model_load(values):
    with pytest.raises(ValueError):
        InferenceConfig(**values)


def test_no_silent_config_typos(tmp_path):
    p=tmp_path/'config.json';p.write_text('{"anchor": 1}')
    with pytest.raises(ValueError,match='Unknown'):
        InferenceConfig.load(p)


def test_prompt_ids_cannot_escape_output_directory(tmp_path):
    p=tmp_path/'prompts.jsonl'
    for sample_id in ['../file','/tmp/file','same/same','']:
        p.write_text(json.dumps(dict(id=sample_id,prompt='test',seed=42)))
        with pytest.raises(ValueError):
            read_prompts(p,None,42)
    record=json.dumps(dict(id='same',prompt='test',seed=42))
    p.write_text(record+'\n'+record)
    with pytest.raises(ValueError,match='Duplicate'):
        read_prompts(p,None,42)


def test_cli_dry_run_needs_no_weights(tmp_path):
    p=subprocess.run([sys.executable,'-m','sdit.cli','generate','--prompt','test',
        '--output',str(tmp_path/'out'),'--dry-run'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['config']['warmup_steps']==6
    assert not (tmp_path/'out').exists()


def test_reset_discards_previous_request_state():
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from sdit.engine import Runner
    from ras.utils import ras_manager
    from ras.modules.attention_processor import RASLuminaAttnProcessor2_0
    processor=RASLuminaAttnProcessor2_0()
    processor.k_cache=torch.ones(1)
    processor.v_cache=torch.ones(1)
    processor.remote_anchor_value_mean_cache=torch.ones(1)
    processor.cache_in_cluster_order=True
    runner=Runner.__new__(Runner)
    runner.config=InferenceConfig()
    runner.pipe=SimpleNamespace(transformer=SimpleNamespace(modules=lambda:[SimpleNamespace(processor=processor)]))
    runner.original_scheduler_config=dict(FlowMatchEulerDiscreteScheduler().config)
    previous=ras_manager.MANAGER
    try:
        first=runner.reset();scheduler=runner.pipe.scheduler
        ras_manager.MANAGER.current_step=27
        scheduler.clustered=True
        scheduler.step_update_history.append(13)
        runner.reset()
        assert ras_manager.MANAGER.current_step==0
        assert runner.pipe.scheduler is not scheduler
        assert not runner.pipe.scheduler.clustered
        assert runner.pipe.scheduler.step_update_history==[]
        assert processor.k_cache is None and processor.v_cache is None
        assert processor.remote_anchor_value_mean_cache is None
        assert not processor.cache_in_cluster_order
        assert first['enable_cluster_attention_density'] is False
        assert first['enable_module_timing'] is False
        assert first['cluster_attention_kernel'] == 'sdpa'
    finally:
        ras_manager.MANAGER=previous


def test_metrics_reject_different_seeds(tmp_path):
    from PIL import Image
    for method,seed in [('sdit',42),('dense',43)]:
        d=tmp_path/method;d.mkdir()
        (d/'run.json').write_text(json.dumps(dict(command='generate',config=InferenceConfig(method=method).to_dict(),
            model='test',prompts=[dict(id='x',prompt='test',seed=seed)])))
        (d/'results.json').write_text(json.dumps([dict(id='x',image='x.png')]))
        Image.new('RGB',(16,16)).save(d/'x.png')
    with pytest.raises(ValueError,match='identical'):
        validate_pair(tmp_path/'sdit',tmp_path/'dense')


def test_evaluate_subcommand_and_legacy_entrypoint_agree(tmp_path):
    from PIL import Image
    for method in ('sdit','dense'):
        directory=tmp_path/method;directory.mkdir()
        (directory/'run.json').write_text(json.dumps(dict(command='generate',
            config=InferenceConfig(method=method).to_dict(),model='test',
            prompts=[dict(id='same',prompt='test',seed=42)])))
        (directory/'results.json').write_text(json.dumps([dict(id='same',image='same.png')]))
        Image.new('RGB',(16,16),(32,64,128)).save(directory/'same.png')
    outputs=[]
    for module,args,name in [('sdit.cli',['evaluate'],'new'),('sdit.evaluate',[],'legacy')]:
        output=tmp_path/(name+'.json')
        result=subprocess.run([sys.executable,'-m',module,*args,'--generated',str(tmp_path/'sdit'),
            '--dense',str(tmp_path/'dense'),'--output',str(output)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        outputs.append(json.loads(output.read_text()))
    assert outputs[0]==outputs[1]
    assert outputs[0]['summary']['ssim']==1
    assert outputs[0]['summary']['psnr_infinite'] is True
