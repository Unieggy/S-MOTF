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


def overfit(steps=1500,batch_size=64):
    """Memorize a single batch; both losses should approach ~0."""
    #extracts one batch and trains it for 300 times

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
    

def train(epochs=20,batch_size=256):
    """full training over synthetic dataset"""
    cfg=load_config()
    device=pick_device()
    print("device:", device)

    episodes=generate_synthetic_data(cfg,n_trajectories=200,trajectory_length=200)
    ds = PlayWindowDataset(episodes, cfg)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SMoTF(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    best=float("inf") # track the lowest total loss

    model.train()

    for epoch in range(epochs):
        running={"total": 0.0, "fm": 0.0, "dyn": 0.0}
        #[256,dim]
        for batch in loader:

            batch=to_device(batch,device)
            #forward pass, calculate the loss
            total,comps=model.loss(batch)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            opt.step()
            sched.step()

            #add the scalar float components to the running trackers
            for k in running:
                running[k]+=comps[k]
        running = {k: v / len(loader) for k, v in running.items()}
        print(f"epoch {epoch:3d} | total {running['total']:.5f} "
              f"| fm {running['fm']:.5f} | dyn {running['dyn']:.5f}")
        if running["total"] < best:
            best = running["total"]
            torch.save(model.state_dict(), "checkpoint_best.pt")


def train_real(epochs=30, batch_size=256, data="go1_data.npz"):
    """Behavior-clone the Go1 teacher on recorded real data (with normalization)."""
    from smotf.data.go1 import load_go1_episodes
    from smotf.data.dataset import compute_norm_stats

    cfg = load_config()
    device = pick_device()
    print("device:", device)

    episodes = load_go1_episodes(data)
    stats = compute_norm_stats(episodes)              # real channels vary wildly -> normalize
    torch.save(stats, "norm_stats.pt")                # reuse identically at rollout
    ds = PlayWindowDataset(episodes, cfg, stats=stats)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SMoTF(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))

    best = float("inf")
    model.train()
    for epoch in range(epochs):
        running = {"total": 0.0, "fm": 0.0, "dyn": 0.0, "align": 0.0}
        for batch in loader:
            batch = to_device(batch, device)
            total, comps = model.loss(batch)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k in running:
                running[k] += comps.get(k, 0.0)
        running = {k: v / len(loader) for k, v in running.items()}
        print(f"epoch {epoch:3d} | total {running['total']:.4f} | fm {running['fm']:.4f} "
              f"| dyn {running['dyn']:.4f} | align {running['align']:.4f}")
        if running["total"] < best:
            best = running["total"]
            torch.save(model.state_dict(), "checkpoint_go1.pt")
    print("saved checkpoint_go1.pt")

if __name__=="__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="train":
        train()
    else:
        overfit()
            

