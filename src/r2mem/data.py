"""Only prepared tensor interfaces; no robot SDK, image decoder or dataset dependency."""
from dataclasses import dataclass
import importlib

import torch


@dataclass
class VisualEpisode:
    episode_id: str
    visuals: torch.Tensor
    mask: torch.Tensor
    times: list[int]

    def validate(self, config):
        if not isinstance(self.episode_id,str) or not self.episode_id:
            raise ValueError('Every episode needs a stable, nonempty episode_id')
        if (self.visuals.ndim != 3 or self.visuals.shape[0] != len(self.times)
                or self.visuals.shape[-1] != config.prefix_dim
                or self.mask.shape != self.visuals.shape[:2] or self.mask.dtype != torch.bool
                or self.visuals.requires_grad or self.mask.device != self.visuals.device or not self.times):
            raise ValueError('VisualEpisode requires detached [T,N,prefix_dim] tokens and a boolean [T,N] mask')
        if (any(type(time) is not int or time < 0 for time in self.times)
                or any(right <= left for left,right in zip(self.times,self.times[1:]))):
            raise ValueError('Observation times must be nonnegative and strictly increasing')


def load_adapter(spec, config):
    if not isinstance(spec,str) or ':' not in spec:
        raise ValueError('Set adapter to a Python factory, e.g. my_adapter:build_adapter')
    module, function = spec.split(':',1)
    adapter = getattr(importlib.import_module(module),function)(config)
    if isinstance(adapter,torch.nn.Module):
        raise TypeError('The adapter must be a plain provider object; return the action model from build_backbone()')
    for name in ('memory_config','memory_episodes','fingerprint'):
        if not callable(getattr(adapter,name,None)):
            raise TypeError(f'Adapter is missing {name}()')
    fingerprint = adapter.fingerprint()
    if not isinstance(fingerprint,str) or not fingerprint:
        raise ValueError('Adapter fingerprint must identify data, splits, preprocessing and the base model')
    return adapter


def prepare_features(memory, adapter, *, fit_frames=16384):
    """Fit the fixed teacher on TRAIN only; keep compact descriptors in CPU RAM."""
    records, seen = {'train':[], 'validation':[]}, {}
    device = next(memory.parameters()).device
    for split in records:
        for episode in adapter.memory_episodes(split):
            episode.validate(memory.config)
            if episode.episode_id in seen:
                raise ValueError(f'Duplicate episode or split leakage: {episode.episode_id}')
            seen[episode.episode_id] = split
            with torch.no_grad():
                raw, valid = memory.teacher._project(episode.visuals.to(device),episode.mask.to(device))
            records[split].append(dict(episode_id=episode.episode_id,times=list(episode.times),
                                       raw=raw.cpu(),valid=valid.cpu()))
        if not records[split]:
            raise ValueError(f'No {split} episodes were supplied')
    raw = torch.cat([record['raw'] for record in records['train']])
    valid = torch.cat([record['valid'] for record in records['train']])
    if len(raw) > fit_frames:
        selected = torch.randperm(len(raw),generator=torch.Generator().manual_seed(memory.config.teacher_seed))[:fit_frames]
        raw, valid = raw[selected],valid[selected]
    memory.teacher.fit_normalization(raw.to(device),valid.to(device))
    for split in records:
        for record in records[split]:
            record['features'] = memory.teacher.normalize_raw(record.pop('raw').to(device),
                                                              record.pop('valid').to(device)).cpu()
    return records
