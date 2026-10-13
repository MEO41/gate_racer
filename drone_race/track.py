"""Race track definition: square gates with position + yaw.

The default layout loosely follows the 7-gate figure-eight track from
Kaufmann et al., Nature 2023 (Fig. 4e), including the split-S
(gates 4 and 5 stacked vertically, flown in opposite directions).
"""
import numpy as np

# Gate geometry (metres). Inner opening is what the drone must pass through.
GATE_INNER = 1.5   # inner opening side length
GATE_OUTER = 2.1   # outer frame side length


class Track:
    def __init__(self, gates):
        """gates: list of (x, y, z, yaw_deg). yaw = direction the drone flies through."""
        g = np.asarray(gates, dtype=np.float64)
        self.pos = g[:, :3]
        yaw = np.deg2rad(g[:, 3])
        self.normal = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], 1)
        self.right = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)], 1)
        self.up = np.tile([0.0, 0.0, 1.0], (len(g), 1))
        self.n = len(g)
        h = GATE_INNER / 2
        # 4 inner corners per gate, world frame: (n_gates, 4, 3)
        offs = [(-h, -h), (h, -h), (h, h), (-h, h)]
        self.corners = np.stack(
            [self.pos + a * self.right + b * self.up for a, b in offs], 1)

    @classmethod
    def figure_eight(cls):
        return cls([
            # x,    y,    z,   yaw
            (0.0,  0.0,  2.0,   45.0),   # 1
            (10.0, 5.5,  2.0,  -60.0),   # 2
            (9.0, -3.5,  2.0, -150.0),   # 3
            (-4.5, -6.0, 3.6,  180.0),   # 4  split-S top
            (-4.5, -6.0, 1.3,    0.0),   # 5  split-S bottom (reverse)
            (4.0, -1.5,  2.0,   60.0),   # 6
            (0.5,  5.0,  2.0,  180.0),   # 7
        ])

    @classmethod
    def circle(cls, n=6, radius=8.0, z=2.0):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return cls([(radius * np.cos(t), radius * np.sin(t), z,
                     np.rad2deg(t) + 90.0) for t in a])
