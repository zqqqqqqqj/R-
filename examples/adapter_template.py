"""Copy to r2mem_adapter.py and connect your existing dataset/model code.

No environment installation, robot driver, raw data format, camera mapping or
action convention is selected by this template. See synthetic.py for a complete
runnable implementation of exactly this interface.
"""
import hashlib
import json

from r2mem import ActionSample
from r2mem.data import VisualEpisode


class YourAdapter:
    def __init__(self,config):
        self.config = config
        # Initialize your existing data loader and its episode-level split here.

    def memory_config(self):
        # Return dimensions from YOUR frozen visual encoder / proprioception.
        # Required keys: prefix_dim, state_dim, camera_indices, coarse_dim.
        # Also specify image_grid if patch-token grids differ from 16x16.
        # coarse_dim must be divisible by 4 * len(camera_indices).
        raise NotImplementedError('Return dimensions from your existing data/model configuration')

    def fingerprint(self):
        # Include actual data/split/normalizer/vision/backbone revisions. Keep this
        # identifier identical between stage 1 and stage 2 of one experiment.
        identity = self.config.get('adapter_options',{}).get('identity')
        required = {'data_revision','split_revision','preprocessing_revision','base_model_revision'}
        if (not isinstance(identity,dict) or not required <= identity.keys()
                or any(not identity[key] for key in required)):
            raise ValueError('Set adapter_options.identity with actual input/base-model revisions')
        return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

    def memory_episodes(self,split):
        # Return a reusable sequence of VisualEpisode objects:
        #   episode_id: stable identity; train and validation must be disjoint
        #   visuals: detached [T, camera_count*image_grid**2, prefix_dim]
        #   mask: boolean [T, token_count]
        #   times: observed timestamps, strictly increasing within this episode
        # These are YOUR existing frozen-vision embeddings, not raw images.
        raise NotImplementedError('Connect your episode-preserving frozen visual feature reader')

    def build_backbone(self,*,device):
        # Load your own pretrained PyTorch action model. Configure its existing
        # trainable parameter selection (full/nonvisual/LoRA) here. Freeze vision.
        raise NotImplementedError('Return your existing configured torch.nn.Module on device')

    def action_sample_count(self,split):
        raise NotImplementedError('Return the number of prepared action decisions in this split')

    def prepare_action_sample(self,index,memory,backbone,*,split,device):
        # Build your existing current prefix, language embeddings and targets.
        # Retain prefix/language gradients to the trainable action model.
        # Use the SAME fitted fixed teacher from stage 1 for the causal history:
        #   features = memory.visual(past_visual_tokens, past_visual_mask)
        # Include the COMPLETE observed episode prefix through current_time,
        # and no future visual observations. memory.visual returns detached FP32.
        # Return ActionSample(...); see synthetic.py for all fields.
        raise NotImplementedError('Return the prepared current decision and complete causal feature prefix')

    def action_loss(self,backbone,sample,augmented_prefix):
        # Run YOUR existing action objective using augmented_prefix, which
        # already contains separate coarse/residual memory tokens and masks.
        # Keep the native state/action transforms, horizon, padding/loss masks
        # and output semantics unchanged. Return a differentiable scalar.
        # For an OpenPI-style embed_prefix API, see openpi_loss.py.
        raise NotImplementedError('Connect the augmented prefix to your native scalar action loss')


def build_adapter(config):
    return YourAdapter(config)
