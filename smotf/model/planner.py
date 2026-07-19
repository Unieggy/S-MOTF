"""the planner (MPC): the flow head PROPOSES, the world model DISPOSES.

At each control step:
  1. sample N candidate actions from the flow head (its sampling noise -> diversity)
  2. score each by rolling the world model H steps and summing predicted reward
  3. execute the FIRST action of the best candidate, then replan next step

Everything runs in NORMALIZED space (the space the world model was trained in):
  state s = [base, legs, contacts] (normalized) -> [N, 40]
  actions from smotf.act() are normalized; reward head returns raw reward.
The returned action is normalized — the caller denormalizes before env.step.
"""
import torch

class Planner:
    def __init__(self,smotf,world,reward,cfg,N=32,H=3):
        self.smotf, self.world, self.reward = smotf, world, reward   # the 3 trained modules
        self.cfg,self.N,self.H=cfg,N,H #N candidates, H horizon 
        # sizes used to slice a 40-d state back into tokens: base=12, legs=24, contacts=4
        self.db, self.dl, self.dc = cfg.dims.base, cfg.dims.legs, cfg.dims.contacts
    def _state_to_context(self,s,command):
        """splits a N,40 state vector back into the token dict the policy can executes"""

        b,l=self.db,self.dl 
        return {
            "base":s[:,:b], #N,12
            "legs":s[:,b:b+l],#N,24
            "contacts":s[:,b+l:b+l+self.dc], #N,4
            "command":command,
        }
    @torch.no_grad()
    def plan(self,context):
        """context: normalized dict, batch size 1
             base[1,12], legs[1,24], contacts[1,4], command[1,C]
        Returns the best first action [12] (normalized)."""
        # build the 40 dim sstate, make N identical copies to image in parallel
        s0=torch.cat([context["base"],context["legs"],context["contacts"]],dim=-1)#1,40
        s=s0.expand(self.N,-1).contiguous() #[1,40] to [N,40]
        cmd=context["command"].expand(self.N,-1).contiguous() #[N,c]

        first=None #hold candidate's first action N,12
        score=torch.zeros(self.N,device=s.device) #N, running sum of predicted reward per candidate

        #roll H steps
        for k in range(self.H):
            ctx=self._state_to_context(s,cmd)
            a=self.smotf.act(ctx) #N,12
            score=score+self.reward(s,a,cmd)

            if k==0:
                first=a
            s=self.world(s,a)

        return first[score.argmax()]
    
if __name__ == "__main__":
    # shape/plumbing check with UNTRAINED modules — numbers are meaningless, shapes must be right
    from smotf import load_config
    from smotf.model.smotf import SMoTF
    from smotf.model.world import WorldModel, RewardHead

    cfg = load_config("configs/multiskill.yaml")        # command=6
    smotf  = SMoTF(cfg).eval()
    world  = WorldModel(cfg).eval()
    reward = RewardHead(cfg).eval()
    planner = Planner(smotf, world, reward, cfg, N=32, H=3)

    ctx = {                                             # one fake NORMALIZED observation
        "base":     torch.randn(1, cfg.dims.base),      # [1,12]
        "legs":     torch.randn(1, cfg.dims.legs),      # [1,24]
        "contacts": torch.randn(1, cfg.dims.contacts),  # [1,4]
        "command":  torch.randn(1, cfg.dims.command),   # [1,6]
    }
    a = planner.plan(ctx)
    print("planned action:", tuple(a.shape))            # (12,)
    print("finite:", bool(torch.isfinite(a).all()))     # True

    # sanity: N=1, H=1 collapses to a single act() call (no search) -> still [12]
    p1 = Planner(smotf, world, reward, cfg, N=1, H=1)
    print("N=1,H=1 action:", tuple(p1.plan(ctx).shape)) # (12,)


