"""Fast, vectorized, noise-free quadrotor gate-racing simulator (pure NumPy).

Simulates N drones in parallel. Follows the structure of the Swift paper
(Kaufmann et al., Nature 2023) but stays deliberately clean and simple:
  * rigid-body dynamics with attitude quaternion
  * action = mass-normalised collective thrust + body rates (like Betaflight "acro")
  * first-order low-level response for thrust and body rates, linear drag
  * 31-D observation: pos(3) vel(3) rotmat(9) next-gate corners rel.(12) prev action(4)
  * reward: progress + perception + command smoothness - crash
No sensor noise, no residual models, no rendering in the loop.
"""
import numpy as np

from .track import Track, GATE_INNER, GATE_OUTER

G = 9.81


def quat_mul(a, b):
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def quat_to_rot(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def yaw_quat(yaw):
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1)


class DroneParams:
    mass = 0.87                 # kg (paper: 870 g)
    max_thrust = 35.0           # N  (paper: ~35 N, TWR 4.1)
    max_rate = np.array([10.0, 10.0, 6.0])   # rad/s roll, pitch, yaw
    tau_thrust = 0.03           # s, first-order motor/thrust lag
    tau_rate = 0.03             # s, first-order body-rate tracking (PID + motors)
    drag = np.array([0.35, 0.35, 0.5])       # linear drag per unit mass, body axes
    cam_tilt = np.deg2rad(30.0)  # camera pitched up (FPV style)


class RewardParams:
    # lambda_1..lambda_5 from Extended Data Table 1a
    l_prog = 1.0
    l_perc = 0.02
    l_perc_exp = -10.0
    l_rate = -2e-4
    l_smooth = -1e-4
    crash = 5.0


class RaceSim:
    OBS_DIM = 31
    ACT_DIM = 4

    def __init__(self, num_envs=100, track=None, dt=0.01, substeps=2,
                 max_steps=1500, bounds=((-12, 15), (-12, 12), (0.0, 8.0)),
                 seed=0, drone=DroneParams, reward=RewardParams, perception=False):
        self.N = num_envs
        self.track = track or Track.figure_eight()
        self.dt, self.sub = dt, substeps
        self.max_steps = max_steps
        self.lo = np.array([b[0] for b in bounds], float)
        self.hi = np.array([b[1] for b in bounds], float)
        self.P, self.Rw = drone, reward
        self.rng = np.random.default_rng(seed)
        self.max_c = drone.max_thrust / drone.mass  # m/s^2
        ct, st = np.cos(drone.cam_tilt), np.sin(drone.cam_tilt)
        self.cam_axis = np.array([ct, 0.0, st])     # optical axis in body frame
        n = self.N
        self.p = np.zeros((n, 3)); self.v = np.zeros((n, 3))
        self.q = np.tile([1.0, 0, 0, 0], (n, 1)); self.w = np.zeros((n, 3))
        self.c = np.full(n, G)                      # actual mass-norm thrust
        self.prev_act = np.zeros((n, 4))
        self.gate = np.zeros(n, int)
        self.t = np.zeros(n, int)
        self.laps = np.zeros(n, int)
        self.gates_passed = np.zeros(n, int)
        self.prev_dist = np.zeros(n)
        self.perception = None
        self.reset()
        if perception:
            from .perception import Perception
            self.perception = Perception(self, seed=seed + 1)

    # ------------------------------------------------------------------ reset
    def reset(self, idx=None):
        idx = np.arange(self.N) if idx is None else np.asarray(idx)
        if idx.size == 0:
            return
        k = len(idx)
        tr = self.track
        g = self.rng.integers(0, tr.n, k)
        # start 1-4 m in front of a random gate, small perturbations
        back = self.rng.uniform(1.0, 4.0, k)
        p = tr.pos[g] - back[:, None] * tr.normal[g]
        p += self.rng.uniform(-0.4, 0.4, (k, 3))
        p[:, 2] = np.clip(p[:, 2], 0.5, self.hi[2] - 0.5)
        yaw = np.arctan2(tr.normal[g, 1], tr.normal[g, 0]) + self.rng.uniform(-0.3, 0.3, k)
        speed = self.rng.uniform(0.0, 6.0, k)
        self.p[idx] = p
        self.v[idx] = speed[:, None] * tr.normal[g] + self.rng.normal(0, 0.3, (k, 3))
        self.q[idx] = yaw_quat(yaw)
        self.w[idx] = self.rng.normal(0, 0.1, (k, 3))
        self.c[idx] = G
        self.prev_act[idx] = 0.0
        self.prev_act[idx, 0] = 2 * G / self.max_c - 1  # hover thrust
        self.gate[idx] = g
        self.t[idx] = 0
        self.laps[idx] = 0
        self.gates_passed[idx] = 0
        self.prev_dist[idx] = np.linalg.norm(self.p[idx] - tr.pos[g], axis=1)
        if self.perception is not None:
            self.perception.reset(idx)

    def reset_podium(self, idx=None, start=(-4.0, 5.0, 0.2)):
        """Race start: drone resting on the ground before gate 1."""
        idx = np.arange(self.N) if idx is None else np.asarray(idx)
        self.reset(idx)
        tr = self.track
        self.p[idx] = start
        self.v[idx] = 0; self.w[idx] = 0
        d = tr.pos[0] - np.array(start)
        self.q[idx] = yaw_quat(np.full(len(idx), np.arctan2(d[1], d[0])))
        self.gate[idx] = 0
        self.prev_dist[idx] = np.linalg.norm(self.p[idx] - tr.pos[0], axis=1)

    # ------------------------------------------------------------- dynamics
    def _act_to_cmd(self, a):
        a = np.clip(a, -1.0, 1.0)
        c_cmd = (a[:, 0] + 1.0) * 0.5 * self.max_c
        w_cmd = a[:, 1:] * self.P.max_rate
        return c_cmd, w_cmd

    def _integrate(self, c_cmd, w_cmd, h):
        P = self.P
        self.c += (c_cmd - self.c) * (h / P.tau_thrust)
        self.w += (w_cmd - self.w) * (h / P.tau_rate)
        R = quat_to_rot(self.q)
        v_body = np.einsum('nji,nj->ni', R, self.v)
        a_body = -P.drag * v_body
        a_body[:, 2] += self.c
        acc = np.einsum('nij,nj->ni', R, a_body)
        acc[:, 2] -= G
        self.v += acc * h
        self.p += self.v * h
        # quaternion update, exact exponential of body rates
        ang = np.linalg.norm(self.w, axis=1) * h
        axis = self.w / np.maximum(np.linalg.norm(self.w, axis=1, keepdims=True), 1e-9)
        dq = np.concatenate([np.cos(ang / 2)[:, None], axis * np.sin(ang / 2)[:, None]], 1)
        self.q = quat_mul(self.q, dq)
        self.q /= np.linalg.norm(self.q, axis=1, keepdims=True)

    # ------------------------------------------------------------- gates
    def _gate_events(self, p0, p1):
        """Detect gate passes / frame collisions on segment p0->p1."""
        tr = self.track
        # signed distance to every gate plane: (N, G)
        d0 = np.einsum('ngk,gk->ng', p0[:, None] - tr.pos[None], tr.normal)
        d1 = np.einsum('ngk,gk->ng', p1[:, None] - tr.pos[None], tr.normal)
        cross = (np.sign(d0) != np.sign(d1)) & (d0 != d1)
        s = np.where(cross, d0 / np.where(d0 - d1 == 0, 1, d0 - d1), 0.0)
        x = p0[:, None] + s[..., None] * (p1 - p0)[:, None]          # crossing point
        rel = x - tr.pos[None]
        lat = np.abs(np.einsum('ngk,gk->ng', rel, tr.right))
        ver = np.abs(np.einsum('ngk,gk->ng', rel, tr.up))
        m = np.maximum(lat, ver)
        inside = cross & (m < GATE_INNER / 2)
        frame = cross & (m >= GATE_INNER / 2) & (m < GATE_OUTER / 2)
        forward = d1 > d0  # crossing in flight direction
        nxt = np.arange(tr.n)[None] == self.gate[:, None]
        passed = (inside & forward & nxt).any(1)
        collided = frame.any(1)
        return passed, collided

    # ------------------------------------------------------------- obs
    def observe(self, truth=False):
        """Policy input. With perception on, built from the estimated state, not the truth.
        truth=True gives the same 31 numbers from the true state (privileged critic input)."""
        if self.perception is not None and not truth:
            pr = self.perception
            p, v, R = pr.p_est, pr.v_est, pr.R_vio
        else:
            p, v, R = self.p, self.v, quat_to_rot(self.q)
        corners = self.track.corners[self.gate] - p[:, None]        # (N,4,3) world
        corners = np.einsum('nji,nkj->nki', R, corners)             # -> body frame
        obs = np.concatenate([
            p / 10.0,
            v / 10.0,
            R.reshape(self.N, 9),
            corners.reshape(self.N, 12) / 10.0,
            self.prev_act], 1)
        return obs.astype(np.float32)

    # ------------------------------------------------------------- step
    def step(self, action):
        """Advance all drones one control step (dt). Returns obs, reward, done, info."""
        action = np.asarray(action, np.float64)
        c_cmd, w_cmd = self._act_to_cmd(action)
        p0 = self.p.copy()
        h = self.dt / self.sub
        for _ in range(self.sub):
            self._integrate(c_cmd, w_cmd, h)
        passed, collided = self._gate_events(p0, self.p)
        if self.perception is not None:
            self.perception.step()

        tr, Rw = self.track, self.Rw
        # progress towards the (current) next gate
        dist = np.linalg.norm(self.p - tr.pos[self.gate], axis=1)
        r_prog = Rw.l_prog * (self.prev_dist - dist)
        # advance gate index on pass
        self.gates_passed += passed
        new_gate = (self.gate + passed) % tr.n
        self.laps += passed & (new_gate == 0)
        self.gate = new_gate
        self.prev_dist = np.linalg.norm(self.p - tr.pos[self.gate], axis=1)

        # perception: camera optical axis towards next gate centre
        R = quat_to_rot(self.q)
        cam = R @ self.cam_axis
        to_g = tr.pos[self.gate] - self.p
        cosang = np.einsum('nk,nk->n', cam, to_g) / np.maximum(np.linalg.norm(to_g, axis=1), 1e-6)
        delta = np.arccos(np.clip(cosang, -1, 1))
        r_perc = Rw.l_perc * np.exp(Rw.l_perc_exp * delta ** 4)

        a = np.clip(action, -1, 1)
        r_cmd = Rw.l_rate * np.sum(a[:, 1:] ** 2, 1) + \
            Rw.l_smooth * np.sum((a - self.prev_act) ** 2, 1)
        self.prev_act = a

        out = np.any((self.p < self.lo) | (self.p > self.hi), 1)
        crashed = collided | out | (self.p[:, 2] < 0.0)
        r = r_prog + r_perc + r_cmd - Rw.crash * crashed

        self.t += 1
        timeout = self.t >= self.max_steps
        info = dict(passed=passed, crashed=crashed, timeout=timeout & ~crashed)
        return self.observe(), r.astype(np.float32), crashed, info
