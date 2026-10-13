"""Fly a trained policy and save trajectories to JSON (for viewer.html / render.py).

    python rollout.py --policy runs/ppo/policy --drones 8 --out traj.json
"""
import argparse, json
import numpy as np
import tensorflow as tf
import tensorflow_probability  # noqa: F401  (registers specs for loading policies)
from tf_agents.environments import tf_py_environment

from drone_race.tf_env import DroneRaceEnv


def obs_dim(policy):
    """31 for normal policies, 62 for ones trained with a privileged critic (actor uses the first 31)."""
    sig = policy.signatures['action'].structured_input_signature[1]
    return [v.shape[-1] for k, v in sig.items() if 'observation' in k][0]
from drone_race.track import GATE_INNER, GATE_OUTER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='runs/ppo/policy_best')
    ap.add_argument('--drones', type=int, default=8)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--laps', type=int, default=3)
    ap.add_argument('--podium', action='store_true', help='race start from the ground')
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--perception', action='store_true',
                    help='policy flies on the VIO + gate-detector estimate; records detections')
    ap.add_argument('--out', default='traj.json')
    a = ap.parse_args()

    pol = tf.saved_model.load(a.policy)
    py = DroneRaceEnv(num_envs=a.drones, max_steps=10**9, seed=a.seed, perception=a.perception,
                      privileged=obs_dim(pol) == 62)
    env = tf_py_environment.TFPyEnvironment(py)
    t = env.reset()
    sim = py.sim
    if a.podium:
        sim.reset_podium(start=(-4.0, 5.0, 0.3))
    t = t._replace(observation=tf.constant(py._obs()))
    state = pol.get_initial_state(a.drones)

    paths = [[] for _ in range(a.drones)]
    quats = [[] for _ in range(a.drones)]
    alive = np.ones(a.drones, bool)
    gate_times = [[] for _ in range(a.drones)]
    # policy input, minus the parts the viewer rebuilds (rotation matrix from quats, previous
    # action from act). Without perception: velocity/10 (3) + corners/10 (12), position = path/10.
    # With perception: position/10 (3) + velocity/10 (3) + corners/10 (12), all from the estimate.
    obs_log = [[] for _ in range(a.drones)]
    perc = sim.perception
    sigma_log = [[] for _ in range(a.drones)]  # Kalman 3D position uncertainty, sqrt(trace P) (m)
    kf_log = [[] for _ in range(a.drones)]     # 0 nothing, 1 detector fix accepted, 2 rejected
    frames = [[] for _ in range(a.drones)]     # camera frames: [index, [[corner, u, v, conf, flag], ...]]
    last_upd = np.zeros(a.drones, int); last_rej = np.zeros(a.drones, int)
    act_log = [[] for _ in range(a.drones)]    # the 4 numbers it answered with
    gate_log = [[] for _ in range(a.drones)]   # index of the gate it was aiming for
    for k in range(a.steps):
        obs = t.observation.numpy()
        act = pol.action(t, state).action
        act_np = np.clip(act.numpy(), -1, 1)
        for i in np.where(alive)[0]:
            paths[i].append(sim.p[i].round(3).tolist())
            quats[i].append(sim.q[i].round(4).tolist())
            if perc is None:
                obs_log[i].append(obs[i, 3:6].round(3).tolist() + obs[i, 15:27].round(3).tolist())
            else:
                obs_log[i].append(obs[i, 0:6].round(3).tolist() + obs[i, 15:27].round(3).tolist())
                sigma_log[i].append(round(float(np.sqrt(np.trace(perc.Pk[i, :3, :3]))), 3))
            act_log[i].append(act_np[i].round(3).tolist())
            gate_log[i].append(int(sim.gate[i]))
        t = env.step(act)
        info = py.info
        if perc is not None:
            for i in np.where(alive)[0]:
                kf_log[i].append(1 if perc.n_updates[i] > last_upd[i] else (2 if perc.n_rejected[i] > last_rej[i] else 0))
            last_upd[:] = perc.n_updates; last_rej[:] = perc.n_rejected
            cap = perc.last
            if cap is not None and cap['step'] == perc.step_count:
                # flag: 1 detected (u, v = noisy detection), 0 visible but missed, 2 hidden behind another gate
                for i in np.where(alive)[0]:
                    cs = np.where(cap['visible'][i])[0]
                    rows = []
                    for c in cs:
                        det = bool(cap['detected'][i, c])
                        flag = 1 if det else (2 if cap['occluded'][i, c] else 0)
                        uu = cap['u'][i, c] if det else cap['u_true'][i, c]
                        vv = cap['v'][i, c] if det else cap['v_true'][i, c]
                        rows.append([int(c), round(float(uu), 1), round(float(vv), 1),
                                     round(float(cap['conf'][i, c]), 2), flag])
                    frames[i].append([k + 1, rows])
        for i in np.where(alive & info['passed'])[0]:
            gate_times[i].append(round(k * sim.dt, 3))
        alive &= ~info['crashed']
        alive &= sim.laps < a.laps
        if not alive.any():
            break

    n_g = sim.track.n
    laps = []
    for i in range(a.drones):
        gt = gate_times[i]
        lap_t = [gt[j + n_g] - gt[j] for j in range(0, len(gt) - n_g, n_g)]
        laps.append(lap_t)
    best = min([l for L in laps for l in L], default=None)
    print(f'gates passed per drone: {[len(g) for g in gate_times]}')
    print(f'lap times: {[[round(x, 2) for x in L] for L in laps]}  best: {best}')

    tr = sim.track
    json.dump(dict(
        dt=sim.dt,
        gates=[dict(pos=tr.pos[g].tolist(), normal=tr.normal[g].tolist(),
                    corners=tr.corners[g].round(4).tolist()) for g in range(n_g)],
        cam_tilt_deg=float(np.rad2deg(sim.P.cam_tilt)),
        max_thrust=float(sim.max_c), max_rate=sim.P.max_rate.tolist(),
        gate_inner=GATE_INNER, gate_outer=GATE_OUTER,
        paths=paths, quats=quats, gate_times=gate_times, best_lap=best,
        obs=obs_log, act=act_log, target=gate_log,
        perception=None if perc is None else dict(
            img=384, fov_deg=100.0, latency_steps=4, sigma=sigma_log, kf=kf_log, frames=frames),
    ), open(a.out, 'w'))
    print('saved', a.out)


if __name__ == '__main__':
    main()
