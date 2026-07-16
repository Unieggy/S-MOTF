"""train the world model on MULTI-STEP rollouts, and measure imagination.

THE GATE: we roll the world model forward K steps feeding it its OWN predictions,
and print the prediction error at each horizon step. If the error stays BOUNDED
across steps, imagination works and planning (Step 18) is viable. If it EXPLODES
by step 3, stop — the model can't imagine and planning is hopeless.

We test on SYNTHETIC data first (dynamics = a known linear map), so a failure
here is a code bug, not hard robot dynamics. Then repeat on real Go1 data.
"""
import sys
import torch
from torch.utils.data import Dataloader

from smotf import load_config
from smotf.data.dataset import PlayWindowDataset, compute_norm_stats
from smotf.model.world import WorldModel, RewardHead

def masked_mse(pred,target,mask):
    """loss for the multi step rollout"""
    if pred.dim()==2: #B,40
        m=mask.unsqueeze(-1).float() #B to B,1
        return ((pred-target)**2*m).sum()/(m.sum()*pred.shape[-1]+1e-8)
    m=mask.float()
    return ((pred-target)**2*m).sum()/(m.sum()+1e-8)