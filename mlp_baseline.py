"""Naive MLP behavior-cloning baseline (for the README comparison).

A plain state->action MLP trained with MSE on the SAME go1_data.npz + the SAME
normalization as s-motf. No flow matching, no world model, no latent plan.
If s-motf beats this, the MoT + flow-matching design earns its complexity.

Run in nanowm:  python mlp_baseline.py     -> saves checkpoint_mlp.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smotf import load_config
from smotf.data.go1 import load_go1_episodes
from smotf.data.dataset import PlayWindowDataset, compute_norm_stats


class MLPPolicy(nn.Module):
    """state (base+legs+contacts+command) -> action, plain regression."""

    def __init__(self, cfg, hidden=512):
        super().__init__()
        d = cfg.dims
        in_dim = d.base + d.legs + d.contacts + d.command      # 43
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d.action),                       # 12, no final activation
        )

    def forward(self, batch):
        x = torch.cat([batch["base"], batch["legs"], batch["contacts"], batch["command"]], dim=-1)
        return self.net(x)


def train_mlp(epochs=30, batch_size=256, data="go1_data.npz", config=None):
    cfg = load_config(config) if config else load_config()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    episodes = load_go1_episodes(data)
    stats = compute_norm_stats(episodes)
    torch.save(stats, "norm_stats.pt")                          # same stats as s-motf
    ds = PlayWindowDataset(episodes, cfg, stats=stats)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = MLPPolicy(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))

    best = float("inf")
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = F.mse_loss(model(batch), batch["action"])    # <- pure BC MSE
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        running /= len(loader)
        print(f"epoch {epoch:3d} | bc_mse {running:.4f}")
        if running < best:
            best = running
            torch.save(model.state_dict(), "checkpoint_mlp.pt")
    print("saved checkpoint_mlp.pt")


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "go1_data.npz"      # e.g. go1_multiskill.npz
    config = sys.argv[2] if len(sys.argv) > 2 else None              # e.g. configs/multiskill.yaml
    train_mlp(data=data, config=config)
