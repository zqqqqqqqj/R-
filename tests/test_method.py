import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file,load_model
from torch import nn
import yaml

from r2mem import MemoryConfig,R2Mem,Stage2Model
from r2mem.checkpoint import load_memory,load_action_model,save_weights
from r2mem.data import prepare_features
from r2mem.training import run
from examples.synthetic import SyntheticAdapter

ROOT = Path(__file__).resolve().parents[1]


def settings(tmp_path):
    cfg = yaml.safe_load((ROOT/'examples/synthetic.yaml').read_text())
    path = tmp_path/'settings.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path,cfg


def construct(adapter,cfg):
    torch.manual_seed(cfg['seed'])
    values = adapter.memory_config()
    values.update(cfg['memory'])
    return R2Mem(MemoryConfig(**values))


@pytest.fixture
def pretrained(tmp_path):
    path,cfg = settings(tmp_path)
    result = run('stage1',path,tmp_path/'stage1')
    assert result['validation']['passed']
    return path,cfg,Path(result['final'])


@pytest.mark.parametrize('state_dim',[3,8,14])
@pytest.mark.parametrize('camera_count',[1,2,3])
def test_memory_is_independent_of_robot_state_width_and_camera_count(state_dim,camera_count):
    config = MemoryConfig(prefix_dim=8,state_dim=state_dim,camera_indices=tuple(range(camera_count)),
        coarse_dim=4*camera_count,image_grid=2,hidden_dim=16,query_dim=8,read_dim=4,slots=2,residual_dim=4)
    memory = R2Mem(config)
    visual = torch.randn(10,4*camera_count,8)
    mask = torch.ones(10,4*camera_count,dtype=torch.bool)
    memory.teacher.fit_normalization(*memory.teacher._project(visual,mask))
    features = memory.visual(visual,mask)
    state = memory.replay_features(features,list(range(10)))
    query = memory.current_query(features[-1:],torch.randn(1,3,8),torch.ones(1,3,dtype=torch.bool),
                                 torch.randn(1,state_dim))[0]
    coarse,residual = memory.read(state,query)
    assert coarse.shape == residual.shape == (4,)


def test_teacher_fit_never_uses_validation_features(tmp_path):
    _,cfg = settings(tmp_path)
    adapter = SyntheticAdapter(cfg)
    changed = copy.deepcopy(adapter)
    for episode in changed.episodes['validation']:
        episode.visuals += 1000
    first,second = construct(adapter,cfg),construct(changed,cfg)
    prepare_features(first,adapter,fit_frames=32)
    prepare_features(second,changed,fit_frames=32)
    for key in ('mean','scale','raw_std','fit_count','projection'):
        torch.testing.assert_close(getattr(first.teacher,key),getattr(second.teacher,key),atol=0,rtol=0)


def test_split_leakage_is_rejected(tmp_path):
    _,cfg = settings(tmp_path)
    adapter = SyntheticAdapter(cfg)
    adapter.episodes['validation'][0].episode_id = adapter.episodes['train'][0].episode_id
    with pytest.raises(ValueError,match='split leakage'):
        prepare_features(construct(adapter,cfg),adapter)


def test_action_only_loss_trains_writer_and_adapter_but_not_history_head(pretrained):
    _,cfg,directory = pretrained
    adapter = SyntheticAdapter(cfg)
    memory,_ = load_memory(directory)
    model = Stage2Model(memory,adapter.build_backbone(device='cpu'),adapter,history_weight=0)
    sample = adapter.prepare_action_sample(11,memory,model.backbone,split='train',device='cpu')
    loss,metrics = model(sample)
    loss.backward()
    assert metrics['history_queries'] == 0
    for module in (model.memory.codec.writer,model.memory.decoder,model.receiver,model.backbone):
        assert any(value.grad is not None and value.grad.abs().sum() > 0 for value in module.parameters())
    for module in (memory.time_query,memory.history_head):
        assert all(not value.requires_grad and value.grad is None for value in module.parameters())


def test_method_release_contains_only_portable_weights_and_metadata(pretrained,tmp_path):
    _,_,directory = pretrained
    assert {path.name for path in directory.iterdir()} == {'config.json','memory.safetensors'}
    memory,document = load_memory(directory)
    assert memory.teacher.fitted.item()
    assert 'optimizer' not in document and 'rng' not in document
    weights = directory/'memory.safetensors'
    corrupted = tmp_path/'broken'
    import shutil
    shutil.copytree(directory,corrupted)
    with (corrupted/weights.name).open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        load_memory(corrupted)


def test_stage2_runs_exports_ema_and_loads_without_stage1_directory(pretrained,tmp_path):
    path,cfg,directory = pretrained
    result = run('stage2',path,tmp_path/'stage2',memory_init=directory)
    assert result['validation']['weights'] == 'complete_ema'
    final = Path(result['final'])
    assert {path.name for path in final.iterdir()} == {
        'config.json','memory','receiver.safetensors','backbone.safetensors'}
    model,document = load_action_model(final,SyntheticAdapter(cfg))
    assert document['weights'] == 'complete_ema'
    sample = model.adapter.prepare_action_sample(11,model.memory,model.backbone,split='validation',device='cpu')
    augmented,_ = model.augment_prefix(sample)
    assert augmented[0].shape[1] == sample.prefix.shape[1]+2
    assert all(not value.requires_grad for value in model.parameters())


@pytest.mark.parametrize('stage',['stage1','stage2'])
def test_interrupted_resume_matches_uninterrupted_parameters_and_ema(pretrained,tmp_path,monkeypatch,stage):
    import r2mem.training as training
    path,_,directory = pretrained
    full,resumed = tmp_path/'full',tmp_path/'resumed'
    kwargs = {'memory_init':directory} if stage == 'stage2' else {}
    run(stage,path,full,**kwargs)
    original = training.publish_checkpoint
    class Interrupted(RuntimeError):
        pass
    def interrupt(*args,**kw):
        original(*args,**kw)
        raise Interrupted('after a complete published checkpoint')
    monkeypatch.setattr(training,'publish_checkpoint',interrupt)
    with pytest.raises(Interrupted):
        run(stage,path,resumed,**kwargs)
    with (resumed/'metrics.jsonl').open('a') as stream:
        stream.write(json.dumps({'step':99999,'loss':0})+'\n')
    monkeypatch.setattr(training,'publish_checkpoint',original)
    run(stage,path,resumed,resume=True,**kwargs)
    for first in (full/'final').rglob('*.safetensors'):
        second = resumed/'final'/first.relative_to(full/'final')
        a,b = load_file(str(first)),load_file(str(second))
        assert a.keys() == b.keys()
        for name,value in a.items():
            torch.testing.assert_close(value,b[name],atol=0,rtol=0,msg=name)
    rows = [json.loads(line)['step'] for line in (resumed/'metrics.jsonl').read_text().splitlines()]
    assert rows == list(range(1,yaml.safe_load(path.read_text())[stage]['steps']+1))
    assert list(resumed.glob('discarded_metrics_*.jsonl'))


def test_stage2_rejects_adapter_that_ignores_memory_tokens(pretrained,tmp_path,monkeypatch):
    path,_,directory = pretrained
    def ignore(self,backbone,sample,augmented):
        return (backbone['action'](sample.prefix.mean(1))-sample.payload).square().mean()
    monkeypatch.setattr(SyntheticAdapter,'action_loss',ignore)
    with pytest.raises(ValueError,match='ignored the augmented prefix'):
        run('stage2',path,tmp_path/'bad',memory_init=directory)


def test_tied_native_model_weights_roundtrip_without_duplication(tmp_path):
    def create():
        model = nn.ModuleDict({'embedding':nn.Embedding(10,8),'head':nn.Linear(8,10,bias=False)})
        model['head'].weight = model['embedding'].weight
        return model
    model,restored = create(),create()
    path = tmp_path/'tied.safetensors'
    save_weights(model,path)
    load_model(restored,str(path))
    assert restored['embedding'].weight is restored['head'].weight
    torch.testing.assert_close(model['embedding'].weight,restored['embedding'].weight,atol=0,rtol=0)


def test_stage2_rejects_unaccepted_memory_release(pretrained,tmp_path):
    path,_,directory = pretrained
    document = json.loads((directory/'config.json').read_text())
    document['metadata']['validation']['passed'] = False
    (directory/'config.json').write_text(json.dumps(document))
    with pytest.raises(ValueError,match='validation'):
        run('stage2',path,tmp_path/'bad',memory_init=directory)
