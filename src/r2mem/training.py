"""Reference stage-1/stage-2 trainers for user-owned data and action-model adapters.

The reference loop is single-process, with explicit gradient accumulation. The
method modules can also be used inside the recipient's distributed trainer.
"""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
import yaml

from .checkpoint import load_memory, save_memory, save_weights, sha256
from .config import MemoryConfig
from .data import load_adapter, prepare_features
from .model import R2Mem, Stage2Model
from .ema import ParameterEMA


def load_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg,dict) or cfg.get('format_version') != 1:
        raise ValueError('Expected method configuration format_version=1')
    for key in ('stage1','stage2'):
        settings = cfg[key]
        for name in ('steps','batch_size','save_every','validate_every','validation_samples'):
            if type(settings.get(name)) is not int or settings[name] < 1:
                raise ValueError(f'{key}.{name} must be a positive integer')
    for name in ('lr','max_grad_norm'):
        if not math.isfinite(cfg['stage1'][name]) or cfg['stage1'][name] <= 0:
            raise ValueError(f'stage1.{name} must be positive and finite')
    for name in ('backbone_lr','memory_lr','interface_lr','max_grad_norm'):
        if not math.isfinite(cfg['stage2'][name]) or cfg['stage2'][name] <= 0:
            raise ValueError(f'stage2.{name} must be positive and finite')
    if type(cfg.get('seed')) is not int or type(cfg.get('max_queries')) is not int or cfg['max_queries'] < 1:
        raise ValueError('Set an integer seed and a positive max_queries')
    if type(cfg.get('teacher_fit_frames')) is not int or cfg['teacher_fit_frames'] < 2:
        raise ValueError('teacher_fit_frames must be at least 2')
    gain = cfg['stage1'].get('min_validation_gain',.01)
    weight = cfg['stage2'].get('history_weight',.1)
    if not 0 <= gain < 1 or not math.isfinite(weight) or weight < 0:
        raise ValueError('Invalid memory validation gain or history loss weight')
    cfg['stage2'].setdefault('ema_decay',.999)
    if not 0 <= cfg['stage2']['ema_decay'] < 1:
        raise ValueError('stage2.ema_decay must lie in [0,1)')
    if type(cfg['stage2'].get('warmup_steps')) is not int or cfg['stage2']['warmup_steps'] < 0:
        raise ValueError('stage2.warmup_steps must be a nonnegative integer')
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def source_signature(adapter):
    import inspect
    files = {path.name:sha256(path) for path in Path(__file__).parent.glob('*.py')}
    source = inspect.getsourcefile(type(adapter))
    if source:
        files['adapter_source'] = sha256(source)
    return files


def rng_state():
    values = np.random.get_state()
    return dict(torch=torch.get_rng_state(),python=random.getstate(),
        numpy=(values[0],values[1].tolist(),*values[2:]),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)


def restore_rng(values):
    torch.set_rng_state(values['torch'])
    random.setstate(values['python'])
    state = values['numpy']
    np.random.set_state((state[0],np.asarray(state[1],dtype=np.uint32),*state[2:]))
    if values['cuda'] is not None:
        torch.cuda.set_rng_state(values['cuda'])


def publish_checkpoint(out, step, model, optimizer, metadata, sampler, queries, report, ema=None):
    root = Path(out)/'checkpoints'
    root.mkdir(parents=True,exist_ok=True)
    destination = root/f'step_{step:06d}'
    if destination.exists():
        raise FileExistsError(f'Checkpoint already published: {destination}')
    temporary = Path(tempfile.mkdtemp(prefix='.saving_',dir=root))
    try:
        state = dict(metadata=metadata,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=rng_state(),sampler=sampler.getstate(),queries=queries.get_state(),validation=report,
            trainable_layout=[(name,list(value.shape),str(value.dtype)) for name,value in model.named_parameters()
                              if value.requires_grad])
        state['ema'] = ema.values if ema is not None else None
        torch.save(state,temporary/'trainer.pt')
        (temporary/'complete.json').write_text(json.dumps(dict(step=step))+'\n')
        os.replace(temporary,destination)
        pointer = Path(out)/'latest.json.tmp'
        pointer.write_text(json.dumps(dict(checkpoint=str(destination.relative_to(out))))+'\n')
        pointer.replace(Path(out)/'latest.json')
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
    # Keep the two newest restart points. Weight-only final releases are separate.
    for old in sorted(root.glob('step_*'))[:-2]:
        shutil.rmtree(old)


def restore_checkpoint(out, model, optimizer, metadata, sampler, queries, ema=None):
    checkpoint = Path(out)/json.loads((Path(out)/'latest.json').read_text())['checkpoint']
    marker = json.loads((checkpoint/'complete.json').read_text())
    state = torch.load(checkpoint/'trainer.pt',map_location='cpu',weights_only=True)
    if state['metadata'] != metadata or marker != dict(step=state['step']):
        raise ValueError('Resume configuration, adapter/data identity or source signature changed')
    layout = [(name,list(value.shape),str(value.dtype)) for name,value in model.named_parameters() if value.requires_grad]
    if state['trainable_layout'] != layout:
        raise ValueError('Trainable parameter layout or dtype changed')
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if ema is not None:
        if state.get('ema') is None:
            raise ValueError('Resume checkpoint has no EMA state')
        ema.load(state['ema'])
    sampler.setstate(state['sampler'])
    queries.set_state(state['queries'])
    restore_rng(state['rng'])
    log = Path(out)/'metrics.jsonl'
    if log.is_file():
        rows = log.read_text().splitlines()
        kept,discarded = [],[]
        for row in rows:
            (kept if json.loads(row)['step'] <= state['step'] else discarded).append(row)
        if discarded:
            import time
            (Path(out)/f'discarded_metrics_{time.time_ns()}.jsonl').write_text('\n'.join(discarded)+'\n')
            temporary = log.with_suffix('.jsonl.tmp')
            temporary.write_text('\n'.join(kept)+'\n')
            temporary.replace(log)
    return state['step'],state['validation']


def stage1_validation(memory, records, settings, max_queries, device):
    candidates = [record for record in records if len(record['times']) > memory.config.slots]
    if not candidates:
        raise ValueError('No validation episode is longer than memory capacity')
    sampler, queries = random.Random(863),torch.Generator().manual_seed(864)
    error = energy = 0.
    elements = 0
    with torch.no_grad():
        for _ in range(settings['validation_samples']):
            record = sampler.choice(candidates)
            end = sampler.randrange(memory.config.slots,len(record['times']))+1
            targets,times = record['features'][:end].to(device),record['times'][:end]
            state = memory.replay_features(targets,times)
            _,metrics = memory.history_loss(state,times,targets,max_queries=max_queries,generator=queries)
            error += metrics['delta_sse']
            energy += metrics['target_delta_energy']
            elements += metrics['query_elements']
    valid = elements > 0 and energy > memory.relative_energy_floor*elements
    gain = 1-error/energy if valid else None
    return dict(kind='held_out_memory_reconstruction',query_elements=elements,
        mse=error/elements if elements else None,validation_gain=gain,
        passed=bool(valid and gain >= settings['min_validation_gain']),
        min_validation_gain=settings['min_validation_gain'])


def _action_sample(adapter, model, index, split, device):
    sample = adapter.prepare_action_sample(index,model.memory,model.backbone,split=split,device=device)
    if sample.episode_id not in model.allowed_episode_ids[split]:
        raise ValueError(f'Action sample episode is outside the declared {split} partition')
    return sample


def stage2_validation(model, adapter, settings, device):
    count = adapter.action_sample_count('validation')
    if type(count) is not int or count < 1:
        raise ValueError('No held-out action validation samples')
    was_training = model.training
    previous_rng = rng_state()
    values = []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        model.eval()
        with torch.no_grad(),torch.random.fork_rng(devices=devices):
            seed_all(891)
            for number in range(settings['validation_samples']):
                sample = _action_sample(adapter,model,number % count,'validation',device)
                loss,metrics = model(sample)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite offline validation loss')
                values.append(float(metrics['action_loss']))
    finally:
        model.train(was_training)
        restore_rng(previous_rng)
    return dict(kind='held_out_native_action_loss',action_loss=sum(values)/len(values),
                samples=len(values),physical_robot_rollout=False)


def run(stage, config_path, out, *, memory_init=None, device='cpu', resume=False):
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise ValueError('The reference CLI is single-process; embed R²Mem modules in your own distributed trainer')
    if stage not in {'stage1','stage2'}:
        raise ValueError('Unknown training stage')
    cfg = load_settings(config_path)
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    seed_all(cfg['seed'])
    torch.set_num_threads(4)
    adapter = load_adapter(cfg['adapter'],cfg)
    spec = dict(adapter.memory_config())
    spec.update(cfg.get('memory',{}))
    memory_config = MemoryConfig(**spec)
    settings = cfg[stage]
    fingerprint = adapter.fingerprint()
    pretraining_recipe = dict(seed=cfg['seed'],teacher_fit_frames=cfg['teacher_fit_frames'],
                              max_queries=cfg['max_queries'],settings=cfg['stage1'])
    out = Path(out).resolve()
    if out.exists() and not resume:
        raise FileExistsError('Choose a fresh output directory, or --resume for this exact run')
    if resume and not (out/'latest.json').is_file():
        raise FileNotFoundError('No complete checkpoint to resume')
    sampler,queries = random.Random(cfg['seed']),torch.Generator().manual_seed(cfg['seed']+1)
    metadata = dict(stage=stage,config=cfg,memory_config=memory_config.to_dict(),
                    adapter_fingerprint=fingerprint,source_signature=source_signature(adapter))
    allowed_ids = {split:{episode.episode_id for episode in adapter.memory_episodes(split)}
                   for split in ('train','validation')}
    if (not all(allowed_ids.values()) or allowed_ids['train'] & allowed_ids['validation']):
        raise ValueError('Provide disjoint nonempty train and validation episode partitions')
    split_digests = {split:hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
                     for split,ids in allowed_ids.items()}
    metadata['episode_splits_sha256'] = split_digests
    records = None
    ema = None
    if stage == 'stage1':
        if memory_init is not None:
            raise ValueError('Stage 1 trains fresh memory and takes no --memory-init')
        model = R2Mem(memory_config).to(device)
        records = prepare_features(model,adapter,fit_frames=cfg['teacher_fit_frames'])
        parameters = model.configure_stage1()
        optimizer = torch.optim.AdamW(parameters,lr=settings['lr'],betas=(.9,.95),weight_decay=0.)
        candidates = [record for record in records['train'] if len(record['times']) > memory_config.slots]
        if not candidates:
            raise ValueError('No train episode is longer than memory capacity')
    else:
        if memory_init is None:
            raise ValueError('Stage 2 requires an accepted stage-1 --memory-init weight package')
        memory,release = load_memory(memory_init,device=device)
        if memory.config != memory_config:
            raise ValueError('Stage-1 memory architecture differs from adapter/configuration')
        info = release.get('metadata',{})
        if (info.get('stage') != 'stage1' or not info.get('validation',{}).get('passed')
                or info.get('adapter_fingerprint') != fingerprint
                or info.get('episode_splits_sha256') != split_digests
                or info.get('pretraining_recipe') != pretraining_recipe
                or info.get('step') != cfg['stage1']['steps']):
            raise ValueError('Stage-1 memory validation or data/preprocessing/base-model identity differs')
        gain = info['validation'].get('validation_gain')
        if not isinstance(gain,(int,float)) or not math.isfinite(gain) or gain < cfg['stage1']['min_validation_gain']:
            raise ValueError('Stage-1 held-out gain is below the requested validation threshold')
        metadata['memory_init_sha256'] = sha256(Path(memory_init)/'memory.safetensors')
        metadata['memory_init_config_sha256'] = sha256(Path(memory_init)/'config.json')
        for name in ('build_backbone','action_sample_count','prepare_action_sample','action_loss'):
            if not callable(getattr(adapter,name,None)):
                raise TypeError(f'Stage-2 adapter is missing {name}()')
        backbone = adapter.build_backbone(device=device)
        if not isinstance(backbone,torch.nn.Module):
            raise TypeError('build_backbone() must return a torch.nn.Module')
        model = Stage2Model(memory,backbone,adapter,history_weight=settings['history_weight'],
                            max_queries=cfg['max_queries']).to(device)
        model.allowed_episode_ids = allowed_ids
        pretrained = {id(value) for module in memory.history_train_modules('full') for value in module.parameters()}
        groups = {'backbone':[], 'memory':[], 'interface':[]}
        backbone_ids = {id(value) for value in backbone.parameters()}
        for parameter in model.parameters():
            if parameter.requires_grad:
                group = 'backbone' if id(parameter) in backbone_ids else 'memory' if id(parameter) in pretrained else 'interface'
                groups[group].append(parameter)
        parameters = [value for group in groups.values() for value in group]
        optimizer = torch.optim.AdamW([dict(params=values,lr=settings[name+'_lr'],peak_lr=settings[name+'_lr'])
            for name,values in groups.items() if values],betas=(.9,.95),weight_decay=1e-10)
        ema = ParameterEMA(model,settings['ema_decay'])
        count = adapter.action_sample_count('train')
        if type(count) is not int or count < 1:
            raise ValueError('No action training samples')
    start,report = 0,None
    if resume:
        start,report = restore_checkpoint(out,model,optimizer,metadata,sampler,queries,ema)
    out.mkdir(parents=True,exist_ok=True)
    model.train()
    for step in range(start+1,settings['steps']+1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(settings['batch_size']):
            if stage == 'stage1':
                record = sampler.choice(candidates)
                end = sampler.randrange(memory_config.slots,len(record['times']))+1
                targets,times = record['features'][:end].to(device),record['times'][:end]
                state = model.replay_features(targets,times)
                loss,_ = model.history_loss(state,times,targets,max_queries=cfg['max_queries'],generator=queries)
            else:
                sample = _action_sample(adapter,model,sampler.randrange(count),'train',device)
                loss,_ = model(sample)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            (loss/settings['batch_size']).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(parameters,settings['max_grad_norm'],error_if_nonfinite=True)
        if stage == 'stage2':
            if all(parameter.grad is None for parameter in model.receiver.parameters()):
                raise ValueError('Native action loss ignored the augmented prefix: no gradient reached the memory interface')
            for group in optimizer.param_groups:
                group['lr'] = group['peak_lr']*min(1.,step/max(settings['warmup_steps'],1))
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if step % settings['validate_every'] == 0 or step == settings['steps']:
            with ema.applied(model) if ema is not None else nullcontext():
                report = (stage1_validation(model,records['validation'],settings,cfg['max_queries'],device)
                          if stage == 'stage1' else stage2_validation(model,adapter,settings,device))
                if ema is not None:
                    report['weights'] = 'complete_ema'
        row = dict(step=step,loss=sum(losses)/len(losses))
        if report is not None:
            row['latest_validation'] = report
        with (out/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        if step == start+1 or step % settings.get('log_every',100) == 0 or step == settings['steps']:
            print(json.dumps(row,allow_nan=False),flush=True)
        if step % settings['save_every'] == 0 or step == settings['steps']:
            publish_checkpoint(out,step,model,optimizer,metadata,sampler,queries,report,ema)
    # Publish a weight-only release. Trainer/Adam/RNG files never enter this directory.
    final = out/'final'
    if not final.exists():
        temporary = Path(tempfile.mkdtemp(prefix='.export_',dir=out))
        try:
            if stage == 'stage1':
                save_memory(model,temporary,metadata=dict(stage=stage,step=settings['steps'],
                    adapter_fingerprint=fingerprint,episode_splits_sha256=split_digests,
                    pretraining_recipe=pretraining_recipe,validation=report))
            else:
                with ema.applied(model):
                    save_memory(model.memory,temporary/'memory',metadata=dict(stage='stage2_memory',step=settings['steps']))
                    save_weights(model.receiver,temporary/'receiver.safetensors')
                    save_weights(model.backbone,temporary/'backbone.safetensors')
                document = dict(format_version=1,kind='r2mem_action_policy',step=settings['steps'],
                    adapter_fingerprint=fingerprint,validation=report,weights='complete_ema',
                    receiver_sha256=sha256(temporary/'receiver.safetensors'),
                    backbone_sha256=sha256(temporary/'backbone.safetensors'))
                (temporary/'config.json').write_text(json.dumps(document,indent=2)+'\n')
            os.replace(temporary,final)
        except BaseException:
            shutil.rmtree(temporary,ignore_errors=True)
            raise
    return dict(stage=stage,step=settings['steps'],validation=report,final=str(final))
