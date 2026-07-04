"""Assemble the model and the loss (no latent plan yet).

Wires the pieces from Steps 2-5 into one nn.Module:

    tokenizer  ->  MoT backbone  ->  { dynamics head, velocity head }

Two jobs:
  - forward(batch): draw a flow-matching pair for the batch's clean actions,
    run one pass, return the velocity prediction + dynamics prediction.
  - loss(batch):    L = w_fm * MSE(v, u_target) + w_dyn * MSE(s_hat, s_next).

velocity(a, t, context) is the sampling entry point (used by flow.sample at
deploy). The latent-plan alignment loss (L_align) is added in Step 6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from smotf.model.tokenizer import Tokenizer
from smotf.model.mot import MoTBackbone
from smotf.model.heads import DynamicsHead,VelocityHead
from smotf.model.flow import make_training_pair, sample


class SMoTF(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg

        self.tokenizer=Tokenizer(cfg)
        self.backbone=MoTBackbone(cfg)
        self.dyn_head=DynamicsHead(cfg)
        self.vel_head=VelocityHead(cfg)
    
    def _encode(self,context,a,t):
        """
        Takes raw dict context and aciton a and time t, run the tokenizer and the mot
        """
        H=self.tokenizer(context,a,t) #B,5,d
        return self.backbone(H)
    
    def velocity(self,a,t,context):
        """velocity field for sampling"""

        Z=self._encode(context,a,t) 
        return self.vel_head(Z) #B,action_dim
    
    def forward(self,batch):
        a_clean=batch["action"] #ground truth

        #generate flow matching training pair
        a0,a_t,t,u_target=make_training_pair(a_clean)

        #pass the state dict, the noisy action and time
        Z=self._encode(batch,a_t,t)

        return {
            "v": self.vel_head(Z),             # predicted velocity   [B, action_dim]
            "u_target": u_target,              # target velocity      [B, action_dim]
            "s_hat": self.dyn_head(Z),         # predicted next state [B, base_dim]
            "s_next": batch["s_next"],         # target next state    [B, base_dim]
        }
    
    def loss(self,batch):
        """computes the weighted sum of the flow matching loss and the dynamics(state) loss"""

        out=self.forward(batch)
        w=self.cfg.weights

        #L_fm
        L_fm=F.mse_loss(out["v"],out["u_target"])

        L_dyn=F.mse_loss(out["s_hat"],out["s_next"])

        total=w.fm*L_fm+w.dyn*L_dyn

        components={"total":total.item(),"fm":L_fm.item(),"dyn":L_dyn.item()}

        return total,components
    
    @torch.no_grad()
    def act(self,context,steps=None):
        """
        deploy:generate a clean acion from an obs
        """
        steps=steps or self.cfg.flow_steps
        B=context["base"].shape[0]
        a0 = torch.randn(B, self.cfg.dims.action, device=context["base"].device)# random noise
        return sample(lambda a, t: self.velocity(a, t, context), a0, steps=steps)

if __name__ == "__main__":
    from smotf import load_config
    from smotf.data.synthetic import generate_synthetic_data
    from smotf.data.dataset import PlayWindowDataset
    from torch.utils.data import DataLoader

    cfg = load_config()
    episodes = generate_synthetic_data(cfg, n_trajectories=8, trajectory_length=20)
    ds = PlayWindowDataset(episodes, cfg)
    batch = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))

    model = SMoTF(cfg)

    # --- loss returns a scalar + finite components ---
    total, comps = model.loss(batch)
    print("loss components:", {k: round(v, 4) for k, v in comps.items()})
    print("all finite:", all(torch.isfinite(torch.tensor(v)) for v in comps.values()))

    # --- backward populates grads everywhere (tokenizer, every block, both heads) ---
    total.backward()
    n_params = sum(1 for _ in model.parameters())
    n_no_grad = sum(1 for p in model.parameters() if p.grad is None)
    print(f"params without grad: {n_no_grad} / {n_params}")   # expect 0

    # --- deploy path produces an action of the right shape ---
    a = model.act(batch)
    print("generated action shape:", tuple(a.shape))          # (16, 12)