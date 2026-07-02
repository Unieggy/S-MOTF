"""Flow-matching (rectified flow): interpolation, target, sampler.

Convention: NOISE at t=0, DATA at t=1. A straight path between them:

    a0        ~ N(0, I)                      # noise            (t = 0)
    a_clean   = the real action              # data             (t = 1)
    a_t       = (1 - t) * a0 + t * a_clean   # point on the path at time t
    u_target  = a_clean - a0                 # velocity (CONSTANT along the path)

Training: regress the model's velocity v_theta(a_t, t, context) -> u_target,
using ONE random t per sample.

Sampling: integrate FORWARD from noise at t=0 to data at t=1 with a few Euler
steps:   a <- a + dt * v_theta(a, t) .   The '+' and the t schedule are the part
that must be exactly right — the round-trip test below catches a wrong sign.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_training_pair(a_clean):
    """given clean actions [B,D],build one flow matching training pair
    Returns (a0, a_t, t, u_target):
        a0       [B, D]  sampled noise (t = 0)
        a_t      [B, D]  interpolated point at time t
        t        [B]     one random time per sample ~ U(0, 1)
        u_target [B, D]  target velocity = a_clean - a0 (constant along path)
    """
    a0=torch.randn_like(a_clean)
    t=torch.rand(a_clean.shape[0],device=a_clean.device)
    a_t=(1-t)[:,None]*a0+t[:,None]*a_clean
    u_target=a_clean-a0
    return a0,a_t,t,u_target

@torch.no_grad()
def sample(velocity_fn,a0,steps=3):
    """Integrate the ODE forward from noise a0 (t=0) to data (t=1).

    velocity_fn: callable (a [B,D], t [B]) -> v [B,D].
        In Step 7 you'll pass  lambda a, t: model.velocity(a, t, context).
    a0:    [B, D] starting noise.
    steps: number of forward Euler steps (default 3), at t = 0, 1/3, 2/3.
    """
    a=a0.clone()
    dt=1.0/steps
    for k in range(steps):
        t=torch.full((a.shape[0],),k*dt,device=a.device) #[B] with value of k*dt
        v=velocity_fn(a,t)
        a=a+dt*v
    return a

if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 32, 12
    a_clean = torch.randn(B, D)
    a0_fixed = torch.randn(B, D)            # the noise we will round-trip FROM

    # --- (a) sampler sanity with a PERFECT (constant) velocity field ---
    # If v == u_target exactly, integrating 3 steps must recover a_clean exactly.
    # This tests the sampler's SIGN/DIRECTION with zero training noise.
    perfect = lambda a, t: (a_clean - a0_fixed)
    a_perfect = sample(perfect, a0_fixed, steps=3)
    print("perfect-field round-trip MSE:", F.mse_loss(a_perfect, a_clean).item())  # ~0

    # --- (b) round-trip identity test with a LEARNED velocity field ---
    # Overfit a tiny net so v(a_t, t) ~ a_clean - a0_fixed along the path, then
    # sample from the SAME a0 and check we land on a_clean (MSE < 1e-2).
    net = nn.Sequential(
        nn.Linear(D + 1, 128), nn.SiLU(),
        nn.Linear(128, 128), nn.SiLU(),
        nn.Linear(128, D),
    )
    velocity_fn = lambda a, t: net(torch.cat([a, t[:, None]], dim=-1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    u_target = a_clean - a0_fixed           # constant per sample
    for step in range(2000):
        t = torch.rand(B)
        a_t = (1 - t)[:, None] * a0_fixed + t[:, None] * a_clean   # sweep the path
        loss = F.mse_loss(velocity_fn(a_t, t), u_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("final training loss:", loss.item())

    a_rt = sample(velocity_fn, a0_fixed, steps=3)
    mse = F.mse_loss(a_rt, a_clean).item()
    print(f"learned round-trip MSE: {mse:.4g}", "PASS ✅" if mse < 1e-2 else "FAIL ❌")

    # --- (c) show the WRONG sign diverges (integrating - instead of +) ---
    a = a0_fixed.clone()
    for k in range(3):
        t = torch.full((B,), k / 3)
        a = a - (1 / 3) * velocity_fn(a, t)      # WRONG: minus
    print("wrong-sign round-trip MSE:", F.mse_loss(a, a_clean).item(), "(should be large)")