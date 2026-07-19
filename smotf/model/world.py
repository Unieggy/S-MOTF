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
        sa=torch.cat([s,a],dim=-1) #B,52
        delta=self.net(sa)
        return s+delta #s'=s+delta
    
class RewardHead(nn.Module):
    """r(s, a，command) -> scalar.   [B,40],[B,12],[B,C] -> [B]   (the planner's scoring function)"""

    def __init__(self, cfg, hidden=256):
        super().__init__()
        d_s = state_dim(cfg)                       # 40
        d_a = cfg.dims.action                      # 12
        d_c=cfg.dims.command                        # 6
        self.net = nn.Sequential(
            nn.Linear(d_s + d_a+d_c, hidden),          # 58 -> 256
            nn.SiLU(),                             # 256 -> 256
            nn.Linear(hidden, hidden),             # 256 -> 256
            nn.SiLU(),                             # 256 -> 256
            nn.Linear(hidden, 1),                  # 256 -> 1   (one reward value per sample)
        )

    def forward(self, s, a,c):                       # s: [4,40], a: [4,12]
        sa = torch.cat([s, a,c], dim=-1)             # [4,40]+[4,12] -> [4,52]
        r = self.net(sa)                           # [4,52] -> [4,1]  (trailing size-1 dim)
        return r.squeeze(-1)                        # [4,1] -> [4]     (drop the size-1 dim)
if __name__ == "__main__":
    from smotf import load_config
    cfg = load_config()
    B   = 4
    d_s = state_dim(cfg)                           # 40
    d_a = cfg.dims.action                          # 12
    world, reward = WorldModel(cfg).eval(), RewardHead(cfg).eval()

    s = torch.randn(B, d_s)                        # [4,40]  a fake batch of states
    a = torch.randn(B, d_a)                        # [4,12]  a fake batch of actions
    print("state dim:", d_s, "| action dim:", d_a) # 40 | 12
    print("f(s,a) ->", tuple(world(s, a).shape))   # (4, 40)  <- SAME as input state (closed)
    print("r(s,a) ->", tuple(reward(s, a).shape))  # (4,)     <- one scalar per sample

    # --- chainability test: feed the model's OWN output back 10 steps ---
    s_roll = s                                     # [4,40]  start
    for k in range(10):
        s_roll = world(s_roll, torch.randn(B, d_a))# [4,40] in -> [4,40] out -> becomes next input
    print("10-step rollout finite:", bool(torch.isfinite(s_roll).all()),  # True
          "| shape:", tuple(s_roll.shape))         # (4, 40)  <- still a valid state after 10 hops