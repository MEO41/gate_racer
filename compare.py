"""Fly a policy under different perception conditions and report how it copes.

    python compare.py --policy runs/ppo/policy_best runs/ppo_perc/policy_best
"""
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability  # noqa: F401  (registers specs for loading policies)
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common

from drone_race.tf_env import DroneRaceEnv


def obs_dim(policy):
    """31 for normal policies, 62 for ones trained with a privileged critic (actor uses the first 31)."""
    sig = policy.signatures['action'].structured_input_signature[1]
    return [v.shape[-1] for k, v in sig.items() if 'observation' in k][0]
from drone_race.perception import PerceptionParams

CONDITIONS = {
    'perfect state': dict(perception=False),
    'VIO drift, no detector': dict(perception=True, detector=False),
    'VIO + detector + Kalman': dict(perception=True, detector=True),
}


def run(policy_dir, cond, drones=64, steps=3000, seed=11):
    PerceptionParams.use_detector = cond.get('detector', True)
    pol = tf.saved_model.load(policy_dir)
    py = DroneRaceEnv(num_envs=drones, max_steps=10**9, seed=seed, perception=cond['perception'],
                      privileged=obs_dim(pol) == 62)
    env = tf_py_environment.TFPyEnvironment(py)
    act = common.function(pol.action)
    t = env.reset()
    sim = py.sim
    alive = np.ones(drones, bool)
    gates = np.zeros(drones)
    gate_t = [[] for _ in range(drones)]
    err, det, vis = [], [], []
    for k in range(steps):
        t = env.step(act(t).action)
        info = py.info
        for i in np.where(alive & info['passed'])[0]:
            gate_t[i].append(k * sim.dt)
        gates += info['passed'] & alive
        alive &= ~info['crashed']
        if sim.perception is not None:
            e = np.linalg.norm(sim.perception.p_est - sim.p, axis=1)
            err.append(e[alive])
            c = sim.perception.last
            if c is not None and c['step'] == sim.perception.step_count:
                det.append(c['detected'][alive].sum(1)); vis.append(c['visible'][alive].sum(1))
    n = sim.track.n
    laps = [g[j + n] - g[j] for g in gate_t for j in range(0, len(g) - n, n)]
    out = dict(gates=gates.mean(), crashed=1 - alive.mean(),
               lap_mean=np.mean(laps) if laps else np.nan, lap_best=np.min(laps) if laps else np.nan)
    if err:
        e = np.concatenate(err)
        out.update(err_rms=np.sqrt(np.mean(e ** 2)), err_p95=np.percentile(e, 95))
    if det:
        out.update(detected=np.mean(np.concatenate(det)), visible=np.mean(np.concatenate(vis)))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', nargs='+', required=True)
    ap.add_argument('--drones', type=int, default=64)
    ap.add_argument('--seconds', type=float, default=30)
    a = ap.parse_args()
    print(f'{a.drones} drones, {a.seconds:.0f} s each, flying until first crash')
    print(f'{"policy":34s} {"condition":26s} {"gates":>6s} {"crashed":>8s} {"lap avg":>8s} {"lap best":>9s} '
          f'{"pos err rms":>11s} {"err p95":>8s} {"corners det/vis":>16s}')
    for pdir in a.policy:
        for name, cond in CONDITIONS.items():
            r = run(pdir, cond, a.drones, int(a.seconds * 100))
            dv = f'{r["detected"]:.1f}/{r["visible"]:.1f}' if 'detected' in r else '–'
            er = f'{r["err_rms"]:.2f} m' if 'err_rms' in r else '–'
            e95 = f'{r["err_p95"]:.2f} m' if 'err_p95' in r else '–'
            print(f'{pdir:34s} {name:26s} {r["gates"]:6.1f} {r["crashed"]:8.0%} {r["lap_mean"]:7.2f}s '
                  f'{r["lap_best"]:8.2f}s {er:>11s} {e95:>8s} {dv:>16s}', flush=True)
