"""windowed dataset for training.

Wraps the raw per-episode trajectories from ``generate_synthetic_data`` into a
flat, indexable ``torch.utils.data.Dataset``. On top of the per-timestep fields
it adds the **future window** the posterior encoder needs (Step 6):

    s_future    = base[t+1 : t+1+H]   -> [H, d_base]   (padded near episode ends)
    future_mask = 1 for real steps, 0 for padding       -> [H]  (bool)

Default DataLoader collation stacks each field across the batch, e.g.
``base -> [B, 12]``, ``s_future -> [B, H, 12]``, ``future_mask -> [B, H]``.
"""

import torch
from torch.utils.data import Dataset

_FIELDS=["base","legs","contacts","command","action","s_next"]

def compute_norm_stats(episodes):
    """
    calculates the man and std for every individual feature across the dataset
    """
    stats={}
    for k in _FIELDS:

        #ep[k] is a matrix of shape [trajectory_length,feature_dim]
        #torch.cat stack all the episodes vertically for each modality
        # if trajectory_length = 200, ep["base"] is [200, 12].
        # If we have 100 episodes, allv becomes a matrix of shape [20000, 12].
        allv=torch.cat([ep[k]for ep in episodes])

        #calculate the mean and std along dimension 0
        #collapses 20000,12 to 12
        stats[k]={
            "mean":allv.mean(0),
            "std":allv.std(0)+1e-6
        }

    return stats

class PlayWindowDataset(Dataset):
    """Per-timestep samples + the posterior's future window.

    If ``world_horizon=K`` is given (Phase 2), each sample ALSO carries the
    windows the world model needs to be rolled out K steps:

        state          [40]      full observation at t   (base+legs+contacts)
        a_future       [K, 12]   actions taken at t .. t+K-1  (drive the rollout)
        r_future       [K]       rewards received at t .. t+K-1  (planner objective)
        s_future_full  [K, 40]   full observations at t+1 .. t+K  (rollout targets)
        wm_mask        [K]       True where the step is real, False past the episode end

    Left as None (the default) the dataset behaves exactly as before, so BC
    training is unaffected.
    """

    def __init__(self,episodes,cfg,stats=None,world_horizon=None):
        self.episodes=episodes
        self.H=cfg.future_horizon #steps into the future the posterior looks
        self.d_base=cfg.dims.base #base vector size
        self.stats=stats #normalized dict

        # Phase 2: world-model rollout windows (opt-in)
        self.K=world_horizon
        self.d_action=cfg.dims.action
        self.d_command=cfg.dims.command
        self.d_state=cfg.dims.base+cfg.dims.legs+cfg.dims.contacts   # 40 — the CLOSED state

        #we create a flat index list
        self.index=[]

        #ei=episode index, ep is the dict
        for ei,ep in enumerate(episodes):
            T=ep["base"].shape[0]
            
            #append a tuple for every timestep in the episode
            for t in range(T):
                self.index.append((ei,t))

    def __len__(self):
        return len(self.index)
    
    def _norm(self,key,x):
        "z score normalization"

        if self.stats is None:
            return x
        
        s=self.stats[key]

        return (x-s["mean"])/s["std"]
    
    def __getitem__(self, idx):
        ei,t=self.index[idx]
        ep=self.episodes[ei]

        #ep[k][t] extracts the 1d vector at time t
        #k:base,legs...etc
        sample={k:self._norm(k,ep[k][t]) for k in _FIELDS}

        #future window
        #slice the base array from the next step t+1 up to t+1+H
        future=ep["base"][t+1:t+1+self.H]

        n_valid=future.shape[0]

        #create a container for future window 
        s_future=torch.zeros(self.H,self.d_base)
        s_future[:n_valid]=self._norm("base",future)

        mask=torch.zeros(self.H,dtype=torch.bool)
        
        #set the first n valid steps to true
        mask[:n_valid]=True

        sample["s_future"]=s_future
        sample["future_mask"]=mask

        # ---- Phase 2: world-model rollout windows (only if world_horizon set) ----
        if self.K is not None:
            T=ep["base"].shape[0]
            K=self.K

            sample["state"]=self._full_state(ep,t)                      # [40] at t

            a_future=torch.zeros(K,self.d_action)
            c_future = torch.zeros(K, self.d_command)
            r_future=torch.zeros(K)
            s_future_full=torch.zeros(K,self.d_state)
            wm_mask=torch.zeros(K,dtype=torch.bool)

            for k in range(K):
                # need action[t+k], reward[t+k] and the state they lead to: state[t+k+1]
                if t+k+1 < T:
                    a_future[k]=self._norm("action",ep["action"][t+k])
                    r_future[k]=ep["reward"][t+k].reshape(-1)[0]        # raw reward (not normalized)
                    s_future_full[k]=self._full_state(ep,t+k+1)
                    c_future[k] = self._norm("command", ep["command"][t+k])
                    wm_mask[k]=True

            sample["a_future"]=a_future
            sample["r_future"]=r_future
            sample["s_future_full"]=s_future_full
            sample["wm_mask"]=wm_mask
            sample["c_future"] = c_future

        return sample

    def _full_state(self,ep,i):
        """The CLOSED state the world model lives in: [base, legs, contacts] -> [40]."""
        return torch.cat([
            self._norm("base",     ep["base"][i]),
            self._norm("legs",     ep["legs"][i]),
            self._norm("contacts", ep["contacts"][i]),
        ])

if __name__=="__main__":
    from smotf import load_config
    from smotf.data.synthetic import generate_synthetic_data
    from torch.utils.data import DataLoader

    #load para and generate a small synthetic dataset
    cfg=load_config()
    episodes=generate_synthetic_data(cfg,n_trajectories=8,trajectory_length=20)

    ds=PlayWindowDataset(episodes,cfg)
    

    #grabs 4 random timesteps and stack them
    # so a [12] base vector each batch 4,12 matrix
    #each ds[idx] has  {"base":[12], "action":[12], "s_future":[8,12], "future_mask":[8], ...}
    batch=next(iter(DataLoader(ds,batch_size=4,shuffle=True)))
    
    print("batch shapes:")
    #group the modalities for 4 esamples
    """
    batch = {
    "base":        [4, 12],
    "action":      [4, 12],
    "s_future":    [4, 8, 12],
    "future_mask": [4, 8],
    ...
    }
    """

    for k, v in batch.items():
        print(f"  {k:12} {tuple(v.shape)}  ({v.dtype})")


    # --- mask correctness at episode tails ---
    T = 20
    last = ds[T - 1]          # very last step: no future at all
    second_last = ds[T - 2]   # exactly one valid future step
    print("mask @ last step      :", last["future_mask"].tolist())
    print("mask @ second-to-last :", second_last["future_mask"].tolist())
    print("padded rows are zero  :", bool((last["s_future"] == 0).all()))


        

