"""Phase 1 — train the RL TEACHER (Go1 locomotion) in MuJoCo Playground.

Run this in the `go2-rl` env (jax + mujoco + brax + playground), NOT the smotf env.

MuJoCo Playground ships the Unitree Go1 quadruped (not Go2), which is the same
12-joint quadruped and maps onto smotf's dims exactly (base 12 / legs 24 /
contacts 4 / command 3 / action 12), so it is a drop-in teacher.

This is reinforcement learning: the policy learns to walk FROM SCRATCH by
interacting with the MuJoCo physics and a locomotion reward — there is no
dataset. Brax PPO runs the Go1 in the sim, rewards forward locomotion, and
improves the policy through interaction.

Output: go1_policy.pkl — the trained teacher (params + network config), used in
Phase 2 to record the real demonstration data that s-motf is cloned on.
"""

import functools
import pickle

import jax
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

ENV = "Go1JoystickFlatTerrain"      # flat-ground velocity-commanded Go1 walking


def main():
    print("JAX devices:", jax.devices())          # expect [CudaDevice(id=0)]
    print("Training env:", ENV)

    env = registry.load(ENV)
    ppo_params = dict(locomotion_params.brax_ppo_config(ENV))

    # Playground nests the network sizes under 'network_factory'; brax's ppo.train
    # wants an actual factory callable, so pop it out and build the partial.
    net_cfg = dict(ppo_params.pop("network_factory"))
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward", float("nan"))
        print(f"step {step:>10}  reward {r:8.2f}", flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        network_factory=network_factory,
        progress_fn=progress,
        **ppo_params,
    )

    with open("go1_policy.pkl", "wb") as f:
        pickle.dump({"env": ENV, "params": params, "network_factory": net_cfg}, f)
    print("saved go1_policy.pkl  (the teacher)")


if __name__ == "__main__":
    main()
