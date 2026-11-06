"""R²Mem and its prefix adapter, independent of robot and action-backbone APIs."""
from dataclasses import dataclass

import torch
from torch import nn

from .config import MemoryConfig
from .core import HistoryMemory
from .initialization import initialize_named_linears
from .interface import SeparateMemoryReceiver
from .state import MemoryState

class R2Mem(HistoryMemory):
    def __init__(self, config: MemoryConfig):
        self.config = config
        super().__init__(prefix_dim=config.prefix_dim, state_dim=config.state_dim,
            coarse_dim=config.coarse_dim, residual_dim=config.residual_dim, query_dim=config.query_dim,
            read_dim=config.read_dim, hidden_dim=config.hidden_dim, slots=config.slots,
            image_grid=config.image_grid, camera_indices=config.camera_indices,
            storage_dtype=getattr(torch,config.storage_dtype), merge_rule=config.merge_rule,
            time_scale=config.time_scale, teacher_seed=config.teacher_seed,
            teacher_scale_floor=config.teacher_scale_floor, writer_bound=config.writer_bound,
            relative_energy_floor=config.relative_energy_floor)
        initialize_named_linears(self, 'memory',
            zero_linear_names=('codec.writer.ratio','codec.writer.mix','decoder.gate'))

    def replay_features(self, features, times):
        if (features.ndim != 2 or features.shape != (len(times),self.coarse_dim)
                or features.dtype != torch.float32 or not len(times) or not torch.isfinite(features).all()):
            raise ValueError('Expected a complete, finite FP32 causal prefix [T, coarse_dim]')
        state = MemoryState()
        for value, time in zip(features.detach(), times, strict=True):
            state = self.codec.append(state, value, time)
        return state

    def configure_stage1(self):
        self.requires_grad_(False)
        for module in self.history_train_modules('full'):
            module.requires_grad_(True)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def configure_stage2(self, history_weight=.1):
        self.requires_grad_(True)
        self.teacher.requires_grad_(False)
        self.control_heads.requires_grad_(False)
        self.reconstruction_head.requires_grad_(False)
        if history_weight == 0:
            self.time_query.requires_grad_(False)
            self.history_head.requires_grad_(False)


class MemoryTokenAdapter(SeparateMemoryReceiver):
    def __init__(self, config: MemoryConfig):
        super().__init__(config.read_dim, config.prefix_dim, readout_rms_cap=config.readout_rms_cap,
                         memory_token_rms_cap=config.memory_token_rms_cap)
        initialize_named_linears(self, 'receiver')


@dataclass
class ActionSample:
    """One prepared decision, supplied by the user's data/backbone adapter.

    features is the detached, normalized fixed-teacher prefix from episode start
    through the current observation. language and prefix may retain gradients.
    payload is passed back to the user-owned action-loss function unchanged.
    """
    episode_id: str
    current_time: int
    features: torch.Tensor
    times: list[int]
    prefix: torch.Tensor
    padding: torch.Tensor
    attention: torch.Tensor
    language: torch.Tensor
    language_mask: torch.Tensor
    proprio: torch.Tensor
    payload: object = None


class Stage2Model(nn.Module):
    def __init__(self, memory, backbone, adapter, *, history_weight=.1, max_queries=32):
        super().__init__()
        if history_weight < 0 or not torch.isfinite(torch.tensor(history_weight)):
            raise ValueError('history_weight must be finite and nonnegative')
        self.memory, self.backbone, self.adapter = memory, backbone, adapter
        self.receiver = MemoryTokenAdapter(memory.config).to(next(memory.parameters()).device)
        self.history_weight, self.max_queries = float(history_weight), max_queries
        memory.configure_stage2(history_weight)

    def augment_prefix(self, sample):
        config = self.memory.config
        if (not sample.episode_id or not sample.times or sample.times[-1] != sample.current_time
                or sample.proprio.shape != (1,config.state_dim) or not torch.isfinite(sample.proprio).all()
                or sample.prefix.ndim != 3 or sample.prefix.shape[0] != 1
                or sample.prefix.shape[-1] != config.prefix_dim
                or sample.language.ndim != 3 or sample.language.shape[0] != 1
                or sample.language.shape[-1] != config.prefix_dim
                or sample.padding.shape != sample.prefix.shape[:2]
                or sample.attention.shape != sample.padding.shape
                or sample.language_mask.shape != sample.language.shape[:2]):
            raise ValueError('Prepared sample dimensions differ from MemoryConfig')
        if sample.features.requires_grad:
            raise ValueError('Historical fixed-teacher features must be detached from the visual encoder')
        state = self.memory.replay_features(sample.features, sample.times)
        query = self.memory.current_query(sample.features[-1:], sample.language,
                                         sample.language_mask, sample.proprio)[0]
        coarse, residual = self.memory.read(state, query)
        augmented = self.receiver(sample.prefix,sample.padding,sample.attention,(coarse[None],residual[None]))
        return augmented,state

    def forward(self, sample):
        augmented,state = self.augment_prefix(sample)
        action_loss = self.adapter.action_loss(self.backbone, sample, augmented)
        if not isinstance(action_loss,torch.Tensor) or action_loss.ndim != 0:
            raise ValueError('adapter.action_loss must return its native scalar action objective')
        if torch.is_grad_enabled() and not action_loss.requires_grad:
            raise ValueError('The native action objective is detached; stage 2 requires gradients through memory tokens')
        if self.history_weight:
            history_loss, info = self.memory.history_loss(state,sample.times,sample.features,
                max_queries=self.max_queries, tensor_metrics=True)
        else:
            history_loss, info = action_loss.new_zeros(()), {'history_queries':0}
        loss = action_loss+self.history_weight*history_loss
        return loss, dict(action_loss=action_loss.detach(),history_loss=history_loss.detach(),
                          history_queries=info['history_queries'])
