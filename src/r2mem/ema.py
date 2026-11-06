"""EMA of trainable parameters; fixed teacher buffers are kept unchanged."""
from contextlib import contextmanager

import torch


class ParameterEMA:
    def __init__(self,model,decay=.999):
        self.decay = decay
        self.values = {name:value.detach().float().clone() for name,value in model.named_parameters()
                       if value.requires_grad}

    @torch.no_grad()
    def update(self,model):
        for name,value in model.named_parameters():
            if name in self.values:
                self.values[name].lerp_(value.detach().float(),1-self.decay)

    def load(self,values):
        if values.keys() != self.values.keys():
            raise ValueError('EMA parameter set differs')
        for name,value in values.items():
            self.values[name].copy_(value.to(self.values[name]))

    @contextmanager
    def applied(self,model):
        originals = {}
        with torch.no_grad():
            for name,value in model.named_parameters():
                if name in self.values:
                    originals[name] = value.detach().clone()
                    value.copy_(self.values[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name,value in model.named_parameters():
                    if name in originals:
                        value.copy_(originals[name])
