"""Optional loss hook for an existing PyTorch OpenPI embed_prefix interface.

This imports no OpenPI package and configures no robot or environment.
The adapter supplies already transformed observations/actions/preprocessing.
"""
import torch

from r2mem import injected_prefix


def action_loss(backbone,sample,augmented_prefix):
    payload = sample.payload
    required = {'observation','actions','preprocessing'}
    if not isinstance(payload,dict) or not required <= payload.keys():
        raise ValueError('OpenPI example requires prepared observation, actions and preprocessing in payload')
    prepared_prefix = (sample.prefix,sample.padding,sample.attention)
    def use_augmented(prefix,padding,attention,unused):
        return augmented_prefix
    with injected_prefix(backbone,prepared_prefix,payload['preprocessing'],use_augmented,None):
        with torch.autocast(device_type=sample.prefix.device.type,dtype=torch.bfloat16,
                            enabled=sample.prefix.device.type == 'cuda'):
            per_element = backbone(payload['observation'],payload['actions']).float()
    reducer = payload.get('reduce_action_loss')
    return reducer(per_element) if reducer is not None else per_element.mean()
