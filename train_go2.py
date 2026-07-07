"""Phase 1 — train the RL TEACHER (Go2 locomotion) in MuJoCo Playground.

Run this in the `go2-rl` env (jax + mujoco + brax + playground), NOT the smotf env.

This is reinforcement learning: the policy learns to walk FROM SCRATCH by
interacting with the MuJoCo physics and a locomotion reward — there is no
dataset. Brax PPO runs the Go2 in the sim, rewards forward locomotion, and
improves the policy through interaction.

Output: go2_policy.pkl — the trained teacher, used in Phase 2 to record the
real demonstration data that s-motf (the student) will be behavior-cloned on.

NOTE: MuJoCo Playground's API is version-sensitive. If an import or call below
does not match your installed version, run:
    python -c "import mujoco_playground as mp; print(mp.registry.locomotion())"
and check Playground's example training script, then adjust the marked lines.
"""

import pickle

import jax
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params          # <-- version-sensitive
from brax.training.agents.ppo import train as ppo               # <-- version-sensitive


def pick_go2_env():
    """Auto-detect a Go2 locomotion env, preferring flat-terrain / joystick."""
    envs = [e for e in registry.locomotion() if "o2" in e.lower()]
    if not envs:
        raise SystemExit(
            "No Go2 locomotion env found. Check available envs with:\n"
            '  python -c "import mujoco_playground as mp; print(mp.registry.locomotion())"'
        )
    for key in ("joystickflatterrain", "flat", "joystick"):
        for e in envs:
            if key in e.lower():
                return e
    return envs[0]


def main():
    print("JAX devices:", jax.devices())          # expect [CudaDevice(id=0)] on the A800
    env_name = pick_go2_env()
    print("Training env:", env_name)

    env = registry.load(env_name)
    ppo_cfg = locomotion_params.brax_ppo_config(env_name)   # PPO hyperparams for this env

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward", float("nan"))
        print(f"step {step:>10}  reward {r:8.2f}", flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        progress_fn=progress,
        **ppo_cfg,
    )

    with open("go2_policy.pkl", "wb") as f:
        pickle.dump({"env": env_name, "params": params}, f)
    print("saved go2_policy.pkl  (the teacher)")


if __name__ == "__main__":
    main()
