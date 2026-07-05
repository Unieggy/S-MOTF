"""Prior / posterior latent plan (Play-LMP style).

Two encoders that both output a plan vector z ∈ [B, plan_dim]:

  PriorEncoder     (deploy): sees ONLY current state + command  -> z_prior
  PosteriorEncoder (train) : sees the FUTURE window s_future     -> z_posterior

In training the action head is conditioned on z_posterior (a grounded plan
distilled from what the robot actually did next), and L_align trains the prior
to match it via stop-gradient:  ||z_prior - sg(z_posterior)||^2 .
At deploy the posterior is discarded and only z_prior is used.

The posterior is a causal GRU over the future states; we read the hidden state
at the LAST VALID step (per the mask), so zero-padded tail steps never leak in.
"""

import torch
import torch.nn as nn

class PriorEncoder(nn.Module):
    """z prior from present info only, state+command"""
    #z plan added to the action token after linear project to 256 before the mot backbone

    def __init__(self,cfg):
        super().__init__()
        d=cfg.dims
        #total size of 4 state vectors=43
        in_dim=d.base+d.legs+d.contacts+d.commands
        #simple mlp to map B,43 to B,256 to B,128
        self.net=nn.Sequential(
            nn.Linear(in_dim,cfg.d),
            nn.GELU(),
            nn.Linear(cfg.d,cfg.plan_dim)
        )

    def forward(self,batch):
        x=torch.cat(
            [batch["base"],batch["legs"],batch["contacts"],batch["command"]],
            dim=-1,

        )# glue each together along the last dim so -> B,43
        return self.net(x)