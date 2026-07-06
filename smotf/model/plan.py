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
        #simple mlp to map B,43(cuz only one step so no extra dim in the middle) to B,256 to B,128
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
    
class PosteriorEncoder(nn.Module):
    """z_posterior from the future window (training only). Swappable: replace the
    GRU(RNN) with e.g. a frozen visual encoder if cameras are added later."""
    def __init__(self,cfg):
        super().__init__()
        #gru takes a sequence of 12 base vectors ->256
        self.gru=nn.GRU(cfg.dims.base,cfg.d,batch_first=True)

        #project 256 to 128 plan dim
        self.proj=nn.Linear(cfg.d,cfg.plan_dim)

    def forward(self,s_future,mask):
        """
        s_future:[B,8,12] 8 future steps 12 dim per step
        mask:B,8 true means its a real step, false means padding(out of bound)
        """

        out,_=self.gru(s_future) #out put B,8,256
        
        #squash it down to [B] sum for each epi
        lengths=mask.sum(dim=1)

        #get the last idx for each epi [B]
        last_idx=(lengths-1).clamp(min=0)

        last=out[torch.arange(out.shape[0]),last_idx] #[B,256]

        z=self.proj(last) #B,128

        z=z*(lengths>0).float().unsqueeze(-1)

        return z

if __name__ == "__main__":
    from smotf import load_config
    from smotf.data.synthetic import generate_synthetic_data
    from smotf.data.dataset import PlayWindowDataset
    from torch.utils.data import DataLoader

    cfg = load_config()
    episodes = generate_synthetic_data(cfg, n_trajectories=8, trajectory_length=20)
    ds = PlayWindowDataset(episodes, cfg)
    batch = next(iter(DataLoader(ds, batch_size=64, shuffle=True)))

    prior = PriorEncoder(cfg).eval()
    posterior = PosteriorEncoder(cfg).eval()

    z_prior = prior(batch)
    z_post = posterior(batch["s_future"], batch["future_mask"])
    print("z_prior shape:", tuple(z_prior.shape))      # (64, 128)
    print("z_post  shape:", tuple(z_post.shape))       # (64, 128)

    # --- mask-leak test: corrupt ONLY the padded (masked-out) steps; z must not change ---
    s_future2 = batch["s_future"].clone()
    pad = ~batch["future_mask"]                        # [B, H] True where padding
    s_future2[pad] = torch.randn_like(s_future2[pad])  # garbage into padded slots only
    z_post2 = posterior(s_future2, batch["future_mask"])
    print("padded steps do NOT leak:", torch.allclose(z_post, z_post2, atol=1e-6))

    # --- stop-gradient check for the alignment target ---
    detached = z_post.detach()
    print("sg(z_post) requires_grad:", detached.requires_grad)   # False