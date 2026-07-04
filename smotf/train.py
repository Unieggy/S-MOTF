"""Training loop (overfit-then-scale).

Two entry points:
  overfit(): train on ONE batch for many steps. Both losses must collapse to
             ~0 (the Step-1 noise floor). This proves the model wiring is
             correct BEFORE we add the latent plan (Step 6).
  train():   full training over the synthetic dataset with AdamW + cosine LR,
             grad clipping, and per-component logging.

Device picks cuda -> mps -> cpu. AMP is only enabled on CUDA (fp16); on
mac/mps we train in full precision (small model, so it's fine).
"""

import torch
from torch.utils.data import DataLoader

from smotf import load_config
from smotf.data.synthetic import generate_synthetic_data
from smotf.data.dataset import PlayWindowDataset
from smotf.model.smotf import SMoTF

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def overfit(steps=300,batch_size=64):
    """Memorize a single batch; both losses should approach ~0."""
    cfg=load_config()
    device=pick_device()
    print("device:", device)

    episodes=generate_synthetic_data(cfg,n_trajectories=8,trajectory_length=40)
    ds=PlayWindowDataset(episodes,cfg)
    batch=next(iter(DataLoader(ds,batch_size=batch_size,shuffle=True)))

    batch=to_device(batch,device)

    model=SMoTF(cfg).to(device)

    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr)

    model.train()

    for step in range(steps):
        total,comps=model.loss(batch)

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d} | total {comps['total']:.5f} "
                  f"| fm {comps['fm']:.5f} | dyn {comps['dyn']:.5f}")

    print("\nOVERFIT PASS ✅" if comps["fm"] < 0.05 and comps["dyn"] < 0.05
          else "\nOVERFIT FAIL ❌ (losses did not collapse — check wiring)")