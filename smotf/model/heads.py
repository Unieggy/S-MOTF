"""
Heads: read the backbone output into the two predictions.

The MoT backbone returns Z ∈ [B, 5, d]. Two heads slice specific tokens:

    DynamicsHead: MLP on Z[:, 0]  (base token)   -> s_hat_next  ∈ [B, dims.base]
    VelocityHead: MLP on Z[:, 4]  (action token) -> v_theta     ∈ [B, dims.action]

These are the load-bearing indices from the tokenizer's fixed order
[base, legs, contacts, command, action]. Both are regression outputs, so the
final layer has NO activation.
"""

import torch
import torch.nn as nn

BASE_TOKEN=0 #row 0, state module
ACTION_TOKEN=4 #action module

class _MLPHEAD(nn.Module):
    """small 2 layer mlp """

    def __init__(self,d,out_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,d),
            nn.GELU(),
            nn.Linear(d,out_dim)
        )

    def forward(self,x): #[B,d]
        return self.net(x) #B,outdim
    
class DynamicsHead(nn.Module):
    """predicts the next base state form the base token"""

    def __init__(self,cfg):
        super().__init__()
        self.head=_MLPHEAD(cfg.d,cfg.dims.base)
    def forward(self, Z): #B,5,d
        return self.head(Z[:,BASE_TOKEN]) #[B,dims.base]
    

class VelocityHead(nn.Module):
    """predicts the flow matching velocity field from the action"""

    def __init__(self,cfg):
        super().__init__()
        self.head=_MLPHEAD(cfg.d,cfg.dims.action)
    def forward(self,Z):
        return self.head(Z[:,ACTION_TOKEN]) #[B,dims.action]
    

if __name__=="__main__":
    from smotf import load_config
    cfg=load_config()

    B=4
    Z=torch.randn(B,5,cfg.d)

    dyn=DynamicsHead(cfg).eval()
    vel=VelocityHead(cfg).eval()
    s_hat=dyn(Z)
    v=vel(Z)
    print("s_hat shape:", tuple(s_hat.shape), "| finite:", bool(torch.isfinite(s_hat).all()))
    print("v     shape:", tuple(v.shape),     "| finite:", bool(torch.isfinite(v).all()))

    # sanity: dynamics head reads ONLY the base token, velocity head ONLY the action token
    Z2 = Z.clone()
    Z2[:, ACTION_TOKEN] = torch.randn(B, cfg.d)         # perturb action token only
    print("dyn ignores action token:", torch.allclose(dyn(Z), dyn(Z2)))   # True
    print("vel reacts to action token:", not torch.allclose(vel(Z), vel(Z2)))  # True