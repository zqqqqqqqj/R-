"""Separate capped coarse/residual prefix tokens."""
import math
import torch
from torch import nn

def cap_rms(values, maximum):
    """Apply an upper RMS bound per vector; never enlarge small or zero signals."""
    values = values.float()
    # Scale before the norm so finite, large readouts do not overflow squaring.
    magnitude = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.)
    rms = torch.linalg.vector_norm(values/magnitude, dim=-1, keepdim=True)
    rms = (rms/math.sqrt(values.shape[-1]))*magnitude
    return values*(maximum/rms.clamp_min(maximum))


class SeparateMemoryReceiver(nn.Module):
    """Two named readouts, two independent lifts, no coarse+residual addition."""
    version = 'separate_capped_v2'

    def __init__(self, read_dim, prefix_dim, hidden=128, *, readout_rms_cap=4., memory_token_rms_cap=1.,
                 residual_enabled=True):
        super().__init__()
        for name, value in (('readout_rms_cap', readout_rms_cap), ('memory_token_rms_cap', memory_token_rms_cap)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
        self.readout_rms_cap = float(readout_rms_cap)
        self.memory_token_rms_cap = float(memory_token_rms_cap)
        self.residual_enabled = residual_enabled
        self.coarse_lift = nn.Sequential(nn.Linear(read_dim, hidden), nn.GELU(), nn.Linear(hidden, prefix_dim))
        self.residual_lift = nn.Sequential(nn.Linear(read_dim, hidden, bias=False), nn.GELU(),
                                           nn.Linear(hidden, prefix_dim, bias=False))
        # Retained state keys for explicit backbone-only/architecture compatibility.
        # V2 does not divide by calibration estimates or amplify near-zero signals.
        self.register_buffer('coarse_scale', torch.ones(()))
        self.register_buffer('residual_scale', torch.ones(()))

    def forward(self, prefix, pad_mask, attention_mask, readouts):
        coarse, residual = readouts
        coarse = self.coarse_lift(cap_rms(coarse, self.readout_rms_cap))
        # Leave room for the final BF16/FP32 rounding so the materialized token
        # retains the configured bound, including a learned coarse-lift bias.
        bound = self.memory_token_rms_cap/(1.+torch.finfo(prefix.dtype).eps)
        values = [cap_rms(coarse, bound)]
        if self.residual_enabled:
            residual = self.residual_lift(cap_rms(residual, self.readout_rms_cap))
            values.append(cap_rms(residual, bound))
        tokens = torch.stack(values, dim=1).to(prefix.dtype)
        valid = torch.ones((len(prefix), len(values)), dtype=pad_mask.dtype, device=prefix.device)
        causal = torch.zeros((len(prefix), len(values)), dtype=attention_mask.dtype, device=prefix.device)
        return torch.cat((prefix, tokens), 1), torch.cat((pad_mask, valid), 1), torch.cat((attention_mask, causal), 1)
