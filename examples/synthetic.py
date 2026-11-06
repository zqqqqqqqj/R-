"""Runnable tensor-only adapter showing both stages. No robot-specific assumptions.

    python scripts/train_stage1.py --config examples/synthetic.yaml --out /tmp/r2mem_s1
    python scripts/train_stage2.py --config examples/synthetic.yaml --memory-init /tmp/r2mem_s1/final --out /tmp/r2mem_s2
"""
import hashlib

import torch
from torch import nn

from r2mem import ActionSample
from r2mem.data import VisualEpisode


class SyntheticAdapter:
    def __init__(self, config):
        generator = torch.Generator().manual_seed(97)
        self.episodes = {'train':[],'validation':[]}
        for split in self.episodes:
            for number in range(3 if split == 'train' else 2):
                visuals = torch.randn(12,4,8,generator=generator)
                visuals += torch.arange(12)[:,None,None]/5
                self.episodes[split].append(VisualEpisode(f'{split}-{number}',visuals,
                    torch.ones(12,4,dtype=torch.bool),list(range(0,120,10))))

    def memory_config(self):
        return dict(prefix_dim=8,state_dim=5,camera_indices=(0,),coarse_dim=8,image_grid=2)

    def fingerprint(self):
        return hashlib.sha256(b'r2mem-synthetic-97-five-dimensional-proprio-v1').hexdigest()

    def memory_episodes(self, split):
        return self.episodes[split]

    def build_backbone(self, *, device):
        model = nn.ModuleDict({'language':nn.Embedding(10,8),'action':nn.Linear(8,3)})
        return model.to(device)

    def action_sample_count(self, split):
        return len(self.episodes[split])*12

    def prepare_action_sample(self,index,memory,backbone,*,split,device):
        episode = self.episodes[split][index//12]
        end = index % 12+1
        visuals,mask = episode.visuals[:end].to(device),episode.mask[:end].to(device)
        features = memory.visual(visuals,mask)
        language = backbone['language'](torch.tensor([[1,2,3]],device=device))
        prefix = torch.cat((visuals[-1:],language),dim=1)
        padding = torch.ones(prefix.shape[:2],device=device,dtype=torch.bool)
        return ActionSample(episode_id=episode.episode_id,current_time=episode.times[end-1],
            features=features,times=episode.times[:end],prefix=prefix,padding=padding,
            attention=torch.zeros_like(padding),language=language,
            language_mask=torch.ones(language.shape[:2],dtype=torch.bool,device=device),
            proprio=torch.zeros(1,5,device=device),payload=torch.zeros(1,3,device=device))

    def action_loss(self,backbone,sample,augmented_prefix):
        prefix,padding,_ = augmented_prefix
        pooled = (prefix*padding[...,None]).sum(1)/padding.sum(1,keepdim=True)
        return (backbone['action'](pooled)-sample.payload).square().mean()


def build_adapter(config):
    return SyntheticAdapter(config)
