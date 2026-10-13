"""Imperfect perception for RaceSim, following the structure of Swift (Kaufmann et al., Nature 2023).

Pipeline per drone (all vectorised over N drones):

  VIO        position/velocity estimate that drifts; drift grows with speed and rotation rate
             (motion blur makes visual features hard to track). Small attitude error too.
  Camera     30 Hz, 384x384 px, 100 deg FOV, pitched up 30 deg.
  Detector   for every gate corner in the image: a detection probability ("confidence") that
             falls with distance (small gate in pixels), oblique viewing angle, motion blur and
             closeness to the image border; corners hidden behind another gate's frame are never
             detected. Detected corners get pixel noise, plus rare large outliers. Results arrive
             40 ms after the frame was taken (Swift's U-Net takes 40 ms on the Jetson TX2).
  Pose       camera position from all detected corners at once. Attitude is taken from VIO
             (Swift only corrects translation), so each corner gives a known bearing ray and the
             position is a linear least-squares fit. Covariance comes from the pixel noise.
  Kalman     filter on the VIO drift (position offset + drift velocity), as in the paper's
             Methods; measurements failing a chi-square test are rejected.

The policy then sees the estimate (VIO minus estimated drift), never the true state.
The drift model is a hand-tuned stochastic process, not the Gaussian processes fitted to real
flights in the paper.
"""
import numpy as np

IMG = 384                         # detector input is 384 x 384 in Swift
FOV_DEG = 100.0
F_PX = (IMG / 2) / np.tan(np.deg2rad(FOV_DEG / 2))   # focal length in pixels
CAM_EVERY = (0, 3, 6)             # capture on these steps of every 10 -> 30 Hz at 100 Hz control
LATENCY_STEPS = 4                 # 40 ms


def skew_exp(e):
    """Rotation matrices from rotation vectors, (N,3) -> (N,3,3)."""
    th = np.linalg.norm(e, axis=1, keepdims=True)
    k = e / np.maximum(th, 1e-12)
    K = np.zeros((len(e), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3)[None] + s * K + (1 - c) * K @ K


class PerceptionParams:
    # --- VIO drift (per drone): velocity error is an Ornstein-Uhlenbeck process
    drift_tau = 2.0             # s, how long a velocity error persists
    drift_sigma0 = 0.03         # m/s/sqrt(s) at hover
    drift_k_speed = 0.012       # extra per m/s of speed
    drift_k_rate = 0.03         # extra per rad/s of body rate
    drift_z_scale = 0.5         # vertical drift is smaller (IMU + gravity help)
    att_sigma_deg = 0.4         # stationary attitude error
    att_tau = 3.0
    # --- detector
    p_max = 0.97                # best-case detection probability per corner
    min_gate_px = 18.0          # gate side length (px) at which detection is 50 % likely
    gate_px_width = 6.0
    min_cos_view = 0.22         # |cos| of viewing angle at which detection is 50 % likely
    exposure = 0.006            # s, sets motion-blur length
    blur_px_half = 9.0          # blur length (px) at which detection probability halves
    edge_px = 14.0              # corners closer than this to the border are unreliable
    max_range = 22.0            # m
    sigma_px = 1.5              # pixel noise of a sharp, detected corner
    sigma_px_per_blur = 0.35
    outlier_prob = 0.03
    outlier_sigma_px = 14.0
    # --- Kalman filter on drift [p_d, v_d]
    kf_q_pos = 0.05             # m/sqrt(s)
    kf_q_vel = 0.3              # m/s/sqrt(s)
    kf_assumed_px = 2.5         # pixel noise the filter assumes
    kf_chi2 = 16.27             # 3 dof, 99.9 %
    min_corners = 3
    use_detector = True         # False = VIO only, drift is never corrected


class Perception:
    def __init__(self, sim, params=PerceptionParams, seed=1):
        self.sim, self.P = sim, params
        self.rng = np.random.default_rng(seed)
        n = sim.N
        tr = sim.track
        self.gate_corners = tr.corners.reshape(-1, 3)          # (G*4, 3)
        self.corner_gate = np.repeat(np.arange(tr.n), 4)
        self.corner_id = np.tile(np.arange(4), tr.n)
        t = sim.P.cam_tilt
        z_c = np.array([np.cos(t), 0.0, np.sin(t)])            # optical axis (body frame)
        x_c = np.array([0.0, -1.0, 0.0])                       # image right
        y_c = np.cross(z_c, x_c)                               # image down
        self.R_bc = np.stack([x_c, y_c, z_c], 1)               # camera -> body
        self.dp = np.zeros((n, 3)); self.dv = np.zeros((n, 3)); self.att = np.zeros((n, 3))
        self.x = np.zeros((n, 6)); self.Pk = np.zeros((n, 6, 6))
        self.epoch = np.zeros(n, int)                          # bumps on reset, drops stale results
        self.step_count = 0
        self.pending = []
        self.last = None                                       # most recent capture, for viewers
        self.n_updates = np.zeros(n, int); self.n_rejected = np.zeros(n, int)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, idx=None):
        n = self.sim.N
        idx = np.arange(n) if idx is None else np.asarray(idx)
        if idx.size == 0:
            return
        k = len(idx)
        sc = np.array([1, 1, self.P.drift_z_scale])
        self.dp[idx] = self.rng.normal(0, 0.05, (k, 3)) * sc
        self.dv[idx] = self.rng.normal(0, 0.03, (k, 3)) * sc
        self.att[idx] = self.rng.normal(0, np.deg2rad(self.P.att_sigma_deg), (k, 3))
        self.x[idx] = 0.0
        self.Pk[idx] = np.diag([0.1 ** 2] * 3 + [0.1 ** 2] * 3)
        self.epoch[idx] += 1
        self.n_updates[idx] = 0; self.n_rejected[idx] = 0

    @property
    def R_vio(self):
        from .sim import quat_to_rot
        return quat_to_rot(self.sim.q) @ skew_exp(self.att)

    @property
    def p_vio(self):
        return self.sim.p + self.dp

    @property
    def v_vio(self):
        return self.sim.v + self.dv

    @property
    def p_est(self):
        return self.p_vio - self.x[:, :3]

    @property
    def v_est(self):
        return self.v_vio - self.x[:, 3:]

    # ------------------------------------------------------------------ step
    def step(self):
        """Call once per control step, after the physics."""
        P, dt, sim = self.P, self.sim.dt, self.sim
        n = sim.N
        # true VIO drift evolves
        speed = np.linalg.norm(sim.v, axis=1)
        rate = np.linalg.norm(sim.w, axis=1)
        sig = P.drift_sigma0 + P.drift_k_speed * speed + P.drift_k_rate * rate
        sc = np.array([1, 1, P.drift_z_scale])
        self.dv += -self.dv / P.drift_tau * dt + sig[:, None] * np.sqrt(dt) * self.rng.normal(size=(n, 3)) * sc
        self.dp += self.dv * dt
        a_sig = np.deg2rad(P.att_sigma_deg) * np.sqrt(2 / P.att_tau)
        self.att += -self.att / P.att_tau * dt + a_sig * np.sqrt(dt) * self.rng.normal(size=(n, 3))
        # Kalman predict
        F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
        Q = np.diag([P.kf_q_pos ** 2] * 3 + [P.kf_q_vel ** 2] * 3) * dt
        self.x = self.x @ F.T
        self.Pk = F @ self.Pk @ F.T + Q
        # camera + detector
        self.step_count += 1
        if self.step_count % 10 in CAM_EVERY:
            self.pending.append((self.step_count + LATENCY_STEPS, self._capture()))
        while self.pending and self.pending[0][0] <= self.step_count:
            cap = self.pending.pop(0)[1]
            if P.use_detector:
                self._update(cap)

    # ------------------------------------------------------------------ detector
    def _capture(self):
        P, sim = self.P, self.sim
        from .sim import quat_to_rot
        n = sim.N
        R_wb = quat_to_rot(sim.q)
        R_wc = R_wb @ self.R_bc                                  # camera -> world (true)
        cam = sim.p                                              # camera at the centre of mass
        rel = self.gate_corners[None] - cam[:, None]             # (N, C, 3)
        pc = np.einsum('nji,ncj->nci', R_wc, rel)                # camera frame
        depth = pc[..., 2]
        front = depth > 0.3
        zsafe = np.where(front, depth, 1.0)
        u = IMG / 2 + F_PX * pc[..., 0] / zsafe
        v = IMG / 2 + F_PX * pc[..., 1] / zsafe
        in_img = front & (u >= 0) & (u < IMG) & (v >= 0) & (v < IMG)
        dist = np.linalg.norm(rel, axis=2)
        in_img &= dist < P.max_range
        occluded = self._occluded(cam, rel) & in_img
        # detection probability = confidence
        gate_px = F_PX * 1.5 / zsafe
        p_size = 1 / (1 + np.exp(-(gate_px - P.min_gate_px) / P.gate_px_width))
        tr = sim.track
        normals = tr.normal[self.corner_gate]                    # (C, 3)
        cosv = np.abs(np.einsum('nck,ck->nc', rel, normals)) / np.maximum(dist, 1e-6)
        p_angle = 1 / (1 + np.exp(-(cosv - P.min_cos_view) / 0.07))
        # blur: image motion during exposure from rotation and from translation
        v_c = np.einsum('nji,nj->ni', R_wc, sim.v)               # camera-frame velocity
        w_c = np.einsum('ji,nj->ni', self.R_bc, sim.w)           # camera-frame body rates
        pix_rate_rot = F_PX * np.linalg.norm(w_c[:, :2], axis=1)[:, None] * np.ones_like(u)
        lat_v = np.linalg.norm(v_c[:, None, :2] - v_c[:, None, 2:3] * pc[..., :2] / zsafe[..., None], axis=2)
        pix_rate_trans = F_PX * lat_v / zsafe
        blur = P.exposure * (pix_rate_rot + pix_rate_trans)
        p_blur = 1 / (1 + (blur / P.blur_px_half) ** 2)
        edge = np.minimum.reduce([u, v, IMG - 1 - u, IMG - 1 - v])
        p_edge = np.clip(edge / P.edge_px, 0, 1)
        conf = P.p_max * p_size * p_angle * p_blur * p_edge * in_img * ~occluded
        detected = self.rng.random(conf.shape) < conf
        sig = P.sigma_px + P.sigma_px_per_blur * blur
        noise = self.rng.normal(size=u.shape + (2,)) * sig[..., None]
        outl = self.rng.random(conf.shape) < P.outlier_prob
        noise += outl[..., None] * self.rng.normal(size=u.shape + (2,)) * P.outlier_sigma_px
        ud, vd = u + noise[..., 0], v + noise[..., 1]
        cap = dict(u=ud, v=vd, u_true=u, v_true=v, conf=conf, detected=detected, visible=in_img,
                   occluded=occluded, epoch=self.epoch.copy(), step=self.step_count,
                   p_vio=self.p_vio.copy(), R_vio=self.R_vio)
        self.last = cap
        return cap

    def _occluded(self, cam, rel):
        """True where the line of sight to a corner passes through another gate's frame."""
        tr = self.sim.track
        from .track import GATE_INNER, GATE_OUTER
        # segment cam -> corner, test against every gate plane: (N, C, G)
        g_pos, g_n, g_r, g_u = tr.pos, tr.normal, tr.right, tr.up
        d0 = np.einsum('ngk,gk->ng', cam[:, None] - g_pos[None], g_n)[:, None, :]      # (N,1,G)
        dd = np.einsum('nck,gk->ncg', rel, g_n)                                          # (N,C,G)
        s = -d0 / np.where(np.abs(dd) < 1e-9, 1e-9, dd)
        hit = cam[:, None, None] + s[..., None] * rel[:, :, None]                        # (N,C,G,3)
        hr = hit - g_pos[None, None]
        a = np.abs(np.einsum('ncgk,gk->ncg', hr, g_r))
        b = np.abs(np.einsum('ncgk,gk->ncg', hr, g_u))
        m = np.maximum(a, b)
        on_frame = (m > GATE_INNER / 2 + 0.02) & (m < GATE_OUTER / 2)
        own = self.corner_gate[None, :, None] == np.arange(tr.n)[None, None, :]
        between = (s > 0.0) & (s < 0.995)
        return (on_frame & between & ~own).any(2)

    # ------------------------------------------------------------------ pose + Kalman update
    def _update(self, cap):
        P = self.P
        ok = (cap['epoch'] == self.epoch)
        det = cap['detected'] & ok[:, None]
        cnt = det.sum(1)
        use = cnt >= P.min_corners
        if not use.any():
            return
        idx = np.where(use)[0]
        R_wc_vio = cap['R_vio'][idx] @ self.R_bc
        ray_c = np.stack([(cap['u'][idx] - IMG / 2) / F_PX, (cap['v'][idx] - IMG / 2) / F_PX,
                          np.ones_like(cap['u'][idx])], -1)
        ray_w = np.einsum('nij,ncj->nci', R_wc_vio, ray_c)
        ray_w /= np.linalg.norm(ray_w, axis=2, keepdims=True)
        A = np.eye(3)[None, None] - ray_w[..., :, None] * ray_w[..., None, :]          # (n,C,3,3)
        # weight by 1 / (angular noise * range)^2, range taken from the VIO-based estimate
        p_prior = cap['p_vio'][idx] - self.x[idx, :3]
        rng_ = np.linalg.norm(self.gate_corners[None] - p_prior[:, None], axis=2)
        sig_ang2 = (P.kf_assumed_px / F_PX) ** 2 + np.deg2rad(P.att_sigma_deg) ** 2
        wgt = det[idx] / (sig_ang2 * np.maximum(rng_, 0.5) ** 2)
        M = np.einsum('nc,ncij->nij', wgt, A)
        b = np.einsum('nc,ncij,ncj->ni', wgt, A, self.gate_corners[None].repeat(len(idx), 0))
        good = np.linalg.cond(M) < 1e6
        idx, M, b = idx[good], M[good], b[good]
        if not len(idx):
            return
        cov = np.linalg.inv(M)
        p_meas = np.einsum('nij,nj->ni', cov, b)                 # measured camera position
        z = cap['p_vio'][idx] - p_meas                           # measured drift
        H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
        Pn = self.Pk[idx]
        y = z - self.x[idx, :3]
        S = Pn[:, :3, :3] + cov
        Si = np.linalg.inv(S)
        maha = np.einsum('ni,nij,nj->n', y, Si, y)
        accept = maha < P.kf_chi2
        self.n_rejected[idx[~accept]] += 1
        idx, Pn, y, Si = idx[accept], Pn[accept], y[accept], Si[accept]
        if not len(idx):
            return
        K = Pn @ H.T @ Si                                         # (n,6,3)
        self.x[idx] += np.einsum('nij,nj->ni', K, y)
        self.Pk[idx] = (np.eye(6)[None] - K @ H[None]) @ Pn
        self.n_updates[idx] += 1
