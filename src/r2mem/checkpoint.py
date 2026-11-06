"""Portable weight releases, separate from optional exact-resume training state."""
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, load_model, save_model

from .config import MemoryConfig, WRITER_VERSION, MEMORY_INTERFACE_VERSION
from .model import R2Mem


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def save_weights(module, path):
    # Native language/action backbones may tie token embeddings and lm_head.
    # Preserve those aliases rather than duplicating multi-GB tensors.
    save_model(module,str(path),metadata={'format':'pt'})


def save_memory(memory, directory, *, metadata=None):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a published memory package: {directory}')
    if not memory.teacher.fitted.item():
        raise ValueError('Fit the fixed teacher on training data before exporting memory')
    directory.mkdir(parents=True,exist_ok=True)
    save_weights(memory,directory/'memory.safetensors')
    document = dict(format_version=1,kind='r2mem_memory',memory_config=memory.config.to_dict(),
        writer_version=WRITER_VERSION,interface_version=MEMORY_INTERFACE_VERSION,
        weights_sha256=sha256(directory/'memory.safetensors'),metadata=metadata or {})
    (directory/'config.json').write_text(json.dumps(document,indent=2)+'\n')
    return directory


def load_memory(directory, *, device='cpu'):
    directory = Path(directory)
    document = json.loads((directory/'config.json').read_text())
    if (document.get('format_version') != 1 or document.get('kind') != 'r2mem_memory'
            or document.get('writer_version') != WRITER_VERSION
            or document.get('interface_version') != MEMORY_INTERFACE_VERSION):
        raise ValueError('Incompatible R²Mem memory release format')
    if sha256(directory/'memory.safetensors') != document.get('weights_sha256'):
        raise ValueError('Memory weight checksum mismatch')
    memory = R2Mem(MemoryConfig(**document['memory_config']))
    memory.load_state_dict(load_file(str(directory/'memory.safetensors')),strict=True)
    if not memory.teacher.fitted.item():
        raise ValueError('Released memory has no fitted teacher')
    return memory.to(device),document


def load_action_model(directory, adapter, *, device='cpu'):
    """Restore method weights using the recipient's native model architecture.

    The adapter remains responsible for its robot transforms, normalizers and
    native inference call. This function never connects to a physical robot.
    """
    from .model import Stage2Model
    directory = Path(directory)
    document = json.loads((directory/'config.json').read_text())
    if (document.get('format_version') != 1 or document.get('kind') != 'r2mem_action_policy'
            or document.get('adapter_fingerprint') != adapter.fingerprint()):
        raise ValueError('Action release or adapter identity differs')
    for name in ('receiver','backbone'):
        if sha256(directory/(name+'.safetensors')) != document.get(name+'_sha256'):
            raise ValueError(f'{name} weight checksum mismatch')
    memory,_ = load_memory(directory/'memory',device=device)
    backbone = adapter.build_backbone(device=device)
    model = Stage2Model(memory,backbone,adapter,history_weight=0).to(device)
    load_model(model.backbone,str(directory/'backbone.safetensors'),strict=True,device='cpu')
    load_model(model.receiver,str(directory/'receiver.safetensors'),strict=True,device='cpu')
    model.requires_grad_(False)
    model.eval()
    return model,document
