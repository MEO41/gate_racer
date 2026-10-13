"""TF-Agents wrapper: one batched PyEnvironment that steps all drones at once."""
import numpy as np
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .sim import RaceSim


class DroneRaceEnv(py_environment.PyEnvironment):
    """Batched env (batch_size = num_envs). Finished drones auto-reset on the next step,
    returning a FIRST step, which is the convention TF-Agents drivers expect."""

    def __init__(self, num_envs=100, discount=0.99, privileged=False, **sim_kwargs):
        """privileged=True appends the true-state observation (31) after the policy's own (31).
        Only the critic uses it during training; the actor network slices off the first 31."""
        super().__init__(handle_auto_reset=False)
        self.sim = RaceSim(num_envs=num_envs, **sim_kwargs)
        self.n = num_envs
        self.privileged = privileged
        self._discount = np.float32(discount)
        dim = RaceSim.OBS_DIM * (2 if privileged else 1)
        self._obs_spec = array_spec.ArraySpec((dim,), np.float32, 'observation')
        self._act_spec = array_spec.BoundedArraySpec(
            (RaceSim.ACT_DIM,), np.float32, -1.0, 1.0, 'action')
        self._last = np.ones(self.n, bool)
        self.ep_gates, self.ep_crash = [], []

    def pop_stats(self):
        g, c = self.ep_gates, self.ep_crash
        self.ep_gates, self.ep_crash = [], []
        return (np.mean(g) if g else np.nan, np.mean(c) if c else np.nan, len(g))

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self.n

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._act_spec

    def _obs(self):
        o = self.sim.observe()
        if self.privileged:
            o = np.concatenate([o, self.sim.observe(truth=True)], 1)
        return o

    def _ts(self, obs, reward, step_type, discount):
        return ts.TimeStep(step_type=step_type.astype(np.int32),
                           reward=reward.astype(np.float32),
                           discount=discount.astype(np.float32),
                           observation=obs)

    def _reset(self):
        self.sim.reset()
        self._last[:] = False
        z = np.zeros(self.n, np.float32)
        return self._ts(self._obs(), z, np.full(self.n, ts.StepType.FIRST),
                        np.ones(self.n, np.float32))

    def _step(self, action):
        restart = np.where(self._last)[0]
        _, r, crashed, info = self.sim.step(action)
        timeout = info['timeout']
        last = crashed | timeout
        step_type = np.where(last, ts.StepType.LAST, ts.StepType.MID)
        # crash = terminal (no bootstrap); timeout = truncation (bootstrap)
        discount = np.where(crashed, 0.0, 1.0)
        # drones that ended last step start a fresh episode now (action ignored)
        if restart.size:
            self.sim.reset(restart)
            r[restart] = 0.0
            step_type[restart] = ts.StepType.FIRST
            discount[restart] = 1.0
            last[restart] = False
        # episode stats (gates passed before ending) for logging
        ended = np.where(last)[0]
        if ended.size:
            self.ep_gates.extend(self.sim.gates_passed[ended].tolist())
            self.ep_crash.extend(crashed[ended].tolist())
        self._last = last
        self.info = info
        return self._ts(self._obs(), r, step_type, discount)


def make_tf_env(num_envs=100, **kw):
    return tf_py_environment.TFPyEnvironment(DroneRaceEnv(num_envs=num_envs, **kw))
