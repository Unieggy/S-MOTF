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
from torch.utils.data import DataLoader

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

def train_world(data=None,epochs=50,K=5,batch_size=256,lr=3e-4):
    if data is None:
        cfg=load_config()
        from smotf.data.synthetic import generate_synthetic_data
        episodes=generate_synthetic_data(cfg,n_trajectories=100,trajectory_length=200)
        print("training wm on synthetic data")

    else:
        cfg=load_config("configs/multiskill.yaml")
        from smotf.data.go1 import load_go1_episodes
        episodes=load_go1_episodes(data)
        print(f"training world model on REAL data: {data}")

    stats=compute_norm_stats(episodes) # for each module stacks all epi and normalize
    torch.save(stats,"norm_stats_world.pt")

    ds=PlayWindowDataset(episodes,cfg,stats=stats,world_horizon=K)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,drop_last=True)

    device= torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    world  = WorldModel(cfg).to(device)                        # f(s[B,40], a[B,12]) -> s'[B,40]
    reward = RewardHead(cfg).to(device)                        # r(s[B,40], a[B,12]) -> [B]
    # optimize BOTH modules together; list(...)+list(...) concatenates their parameter lists
    opt = torch.optim.AdamW(list(world.parameters()) + list(reward.parameters()), lr=lr)

    for epoch in range(epochs):
        step_err=[0.0]*K # running state error at each horizon 0 to k-1
        rew_err=0.0 # running reward pred err
        nb=0 # batch cnt

        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}

            s=batch["state"]# b,40
            loss=0.0

            #multi step rollout
            for k in range(K):
                a=batch["a_future"][:,k] #B,12
                m=batch["wm_mask"][:,k] #[B] at k(time step) whether this step is real
                r_hat=reward(s,a) #pred reward
                loss=loss+masked_mse(r_hat,batch["r_future"][:,k],m)

                s=world(s,a)#next state
                tgt = batch["s_future_full"][:, k]      # [B,40]  what REALLY happened at t+k+1
                se = masked_mse(s, tgt, m)              # state error at THIS horizon step
                loss = loss + se
                step_err[k]+=se.item()
             # a separate 1-step reward error just for logging (uses the untouched real start state)
            
            rew_err += masked_mse(reward(batch["state"], batch["a_future"][:, 0]),
                                  batch["r_future"][:, 0], batch["wm_mask"][:, 0]).item()

            opt.zero_grad()                             # clear old gradients
            loss.backward()                             # backprop through the K-step rollout
            opt.step()                                  # update world + reward weights
            nb += 1

        # -------- THE GATE READOUT: per-horizon-step error, averaged over the epoch --------
        if epoch % 5 == 0 or epoch == epochs - 1:
            # step_err[k]/nb = mean state error when imagining k+1 steps ahead
            errs = " ".join(f"{k+1}:{step_err[k]/nb:.4f}" for k in range(K))
            print(f"epoch {epoch:3d} | step-err {errs} | reward_mse {rew_err/nb:.4f}")

    torch.save(world.state_dict(),  "world.pt")                # the trained world model
    torch.save(reward.state_dict(), "reward.pt")               # the trained reward head
    print("saved world.pt, reward.pt")

    # ============================ INTUITION ============================
    # imagine ONE trajectory and print how far it drifts from reality each step
    world.eval()
    b = next(iter(loader)); b = {k: v.to(device) for k, v in b.items()}
    s = b["state"][:1]                                          # [1,40]  a single sample
    print("\nimagined vs real (mean |err| per step):")
    for k in range(K):
        s = world(s, b["a_future"][:1, k])                     # [1,40]  roll forward one step
        err = (s - b["s_future_full"][:1, k]).abs().mean().item()   # mean absolute error vs truth
        print(f"  step {k+1}: {err:.4f}")


if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else None          # pass a .npz to train on REAL data
    train_world(data)