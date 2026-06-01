"""3-DOF computed torque control helpers.

This module follows the standard robot dynamics form:
    tau = M(q) qdd + C(q, qd) qd + G(q) + F(qd)

The implementation is intentionally explicit and readable so the model can be
replaced later with a robot-specific derivation if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Sequence


@dataclass
class JointParams:
    mass: float = 1.0
    length: float = 0.25
    com_distance: float = 0.1
    inertia: float = 0.0
    motor_inertia: float = 0.0
    gear_ratio: float = 1.0

    @property
    def reflected_inertia(self) -> float:
        """Inertia about the joint axis using Huygens–Steiner + motor reflection."""
        return self.inertia + self.mass * (self.com_distance ** 2) + (self.gear_ratio ** 2) * self.motor_inertia


@dataclass
class CTC3Gains:
    kp: tuple[float, float, float]
    kd: tuple[float, float, float]


@dataclass
class CTC3Model:
    joints: tuple[JointParams, JointParams, JointParams]
    gravity: float = 9.81
    coulomb_friction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    viscous_friction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_scale: float = 1.0


def _as_3(seq: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(seq) != 3:
        raise ValueError(f"{name} must have length 3")
    return float(seq[0]), float(seq[1]), float(seq[2])


def _sign(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0


def _mass_matrix(model: CTC3Model, q: tuple[float, float, float]) -> list[list[float]]:
    """Jacobian-derived 3-DOF mass matrix approximation.

    This remains serial-chain/planar in the x-y plane, but is structured to
    align with the general Jacobian formulation:
        M(q) = sum_k (m_k J_Tk^T J_Tk + J_Rk^T I_Ck J_Rk)
    """
    j1, j2, j3 = model.joints
    q1, q2, q3 = q
    c2 = cos(q2)
    c3 = cos(q3)
    c23 = cos(q2 + q3)

    m1, m2, m3 = j1.mass, j2.mass, j3.mass
    l1, l2 = j1.length, j2.length
    r1, r2, r3 = j1.com_distance, j2.com_distance, j3.com_distance
    I1, I2, I3 = j1.reflected_inertia, j2.reflected_inertia, j3.reflected_inertia

    # Direct closed-form mass matrix terms from the reference derivation.
    M11 = I1 + I2 + I3 + m1 * r1 * r1 + m2 * (l1 * l1 + r2 * r2 + 2.0 * l1 * r2 * c2) + m3 * (
        l1 * l1 + l2 * l2 + r3 * r3 + 2.0 * l1 * l2 * c2 + 2.0 * l1 * r3 * c23 + 2.0 * l2 * r3 * c3
    )
    M12 = I2 + I3 + m2 * (r2 * r2 + l1 * r2 * c2) + m3 * (
        l2 * l2 + r3 * r3 + l1 * l2 * c2 + l1 * r3 * c23 + l2 * r3 * c3
    )
    M13 = I3 + m3 * (r3 * r3 + l1 * r3 * c23 + l2 * r3 * c3)
    M22 = I2 + I3 + m2 * r2 * r2 + m3 * (l2 * l2 + r3 * r3 + 2.0 * l2 * r3 * c3)
    M23 = I3 + m3 * (r3 * r3 + l2 * r3 * c3)
    M33 = I3 + m3 * r3 * r3

    return [
        [M11, M12, M13],
        [M12, M22, M23],
        [M13, M23, M33],
    ]


def _coriolis_matrix(model: CTC3Model, q: tuple[float, float, float], q_dot: tuple[float, float, float]) -> list[list[float]]:
    """Coriolis/centrifugal matrix from the closed-form derivation."""
    j1, j2, j3 = model.joints
    _, q2, q3 = q
    dq1, dq2, dq3 = q_dot

    m2, m3 = j2.mass, j3.mass
    l1, l2 = j1.length, j2.length
    r2, r3 = j2.com_distance, j3.com_distance

    s2 = sin(q2)
    s3 = sin(q3)
    s23 = sin(q2 + q3)

    C = [[0.0 for _ in range(3)] for _ in range(3)]

    C[0][0] = -m2 * l1 * r2 * dq2 * s2 - m3 * l1 * l2 * dq2 * s2 - m3 * l1 * r3 * (dq2 + dq3) * s23 - m3 * l2 * r3 * dq3 * s3
    C[0][1] = -m2 * l1 * r2 * (dq1 + dq2) * s2 - m3 * l1 * l2 * (dq1 + dq2) * s2 - m3 * l1 * r3 * (dq1 + dq2 + dq3) * s23 - m3 * l2 * r3 * dq3 * s3
    C[0][2] = -m3 * l1 * r3 * (dq1 + dq2 + dq3) * s23 - m3 * l2 * r3 * (dq1 + dq2 + dq3) * s3

    C[1][0] = m2 * l1 * r2 * dq1 * s2 + m3 * l1 * l2 * dq1 * s2 + m3 * l1 * r3 * dq1 * s23 - m3 * l2 * r3 * dq3 * s3
    C[1][1] = -m3 * l2 * r3 * dq3 * s3
    C[1][2] = -m3 * l2 * r3 * (dq1 + dq2 + dq3) * s3

    C[2][0] = m3 * l1 * r3 * dq1 * s23 + m3 * l2 * r3 * dq2 * s3
    C[2][1] = m3 * l1 * r3 * (dq1 + dq2) * s23 + m3 * l2 * r3 * dq2 * s3
    C[2][2] = 0.0
    return C


def _gravity_vector(model: CTC3Model, q: tuple[float, float, float]) -> tuple[float, float, float]:
    j1, j2, j3 = model.joints
    q1, q2, q3 = q
    g = model.gravity
    m1, m2, m3 = j1.mass, j2.mass, j3.mass
    l1, l2 = j1.length, j2.length
    r1, r2, r3 = j1.com_distance, j2.com_distance, j3.com_distance

    G1 = g * (
        m1 * r1 * cos(q1)
        + m2 * (l1 * cos(q1) + r2 * cos(q1 + q2))
        + m3 * (l1 * cos(q1) + l2 * cos(q1 + q2) + r3 * cos(q1 + q2 + q3))
    )
    G2 = g * (
        m2 * r2 * cos(q1 + q2)
        + m3 * (l2 * cos(q1 + q2) + r3 * cos(q1 + q2 + q3))
    )
    G3 = g * (m3 * r3 * cos(q1 + q2 + q3))
    return G1, G2, G3


def ctc_3dof(
    qd: Sequence[float],
    q: Sequence[float],
    qd_dot: Sequence[float],
    q_dot: Sequence[float],
    qd_ddot: Sequence[float],
    gains: CTC3Gains,
    model: CTC3Model,
) -> tuple[float, float, float]:
    """Return 3 torque commands using coupled 3-DOF CTC."""
    qd = _as_3(qd, "qd")
    q = _as_3(q, "q")
    qd_dot = _as_3(qd_dot, "qd_dot")
    q_dot = _as_3(q_dot, "q_dot")
    qd_ddot = _as_3(qd_ddot, "qd_ddot")

    e = tuple(qd[i] - q[i] for i in range(3))
    de = tuple(qd_dot[i] - q_dot[i] for i in range(3))
    v = tuple(qd_ddot[i] + gains.kd[i] * de[i] + gains.kp[i] * e[i] for i in range(3))

    M = _mass_matrix(model, q)
    C = _coriolis_matrix(model, q, q_dot)
    G = _gravity_vector(model, q)

    tau: list[float] = []
    for i in range(3):
        tau_i = 0.0
        for j in range(3):
            tau_i += M[i][j] * v[j]
            tau_i += C[i][j] * q_dot[j]
        tau_i += G[i]
        tau_i += model.viscous_friction[i] * q_dot[i]
        tau_i += model.coulomb_friction[i] * _sign(q_dot[i])
        tau.append(tau_i * model.torque_scale)
    return tuple(tau)
