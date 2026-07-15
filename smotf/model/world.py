"""the world model: a CLOSED, chainable dynamics model + reward head.

The Phase-1 DynamicsHead : it mapped a 256-d token to a 12-d BASE
state, so its output could never be fed back as its own input. This module is
CLOSED — f(s, a) -> s' with s and s' the SAME 40-d observation (base+legs+
contacts) — so it can be rolled forward indefinitely under hypothetical actions.
That is what lets the planner (Step 18) imagine without the simulator.

Two choices that matter:
  - PREDICT THE DELTA:  s' = s + mlp([s, a]).  Deltas are small and smooth;
    absolute states are not. This dominates multi-step rollout stability.
  - Small MLPs: called N x H times per control step at deploy, so they must be
    cheap. Do NOT reuse the MoT backbone here.
"""
import torch
import torch.nn as nn

def state_dim(cfg):
    d=cfg.dims
    return d.base+d.legs+d.contacts # 40 dim states

class WorldModel(nn.Module):
    """f(s, a) -> s'   predicts the DELTA, so s' = s + Δ.   [B,40],[B,12] -> [B,40]"""
    def __init__(self,cfg,hidden=256):
        super().__init__()
        d_s=state_dim(cfg) #40
        d_a=cfg.dims.action #12
        #mlp input is state and action concat so 52 dim
        self.net=nn.Sequential(
            nn.Linear(d_s+d_a,hidden), #52 -> 256
            nn.SiLU(),
            nn.Linear(hidden,hidden),
            nn.SiLU(),
            nn.Linear(hidden,d_s)# back to 40
        )
    def forward(self,s,a):
        sa=torch.cat([s,a]) #B,52
        delta=self.net(sa)
        return s+delta #s'=s+delta
    
