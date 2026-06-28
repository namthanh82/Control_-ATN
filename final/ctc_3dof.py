"""3-DOF R-R-R Computed Torque Control (CTC).

Cấu hình robot:
    Joint 1 (q1): Revolute - Hip xoay quanh trục Z
    Joint 2 (q2): Revolute - Knee xoay quanh trục Z
    Joint 3 (q3): Revolute - Ankle xoay quanh trục Z

Prismatic joints bên trong mỗi khâu ảnh hưởng đến:
    - l1 = joints[0].length: Chiều dài hip link (350-450 mm từ VL53L0X)
    - l2 = joints[1].length: Chiều dài knee link (350-450 mm từ VL53L0X)
    - lc1 = joints[0].com_distance: Khoảng cách COM hip
    - lc2 = joints[1].com_distance: Khoảng cách COM knee
    - lc3 = joints[2].com_distance: Khoảng cách COM ankle

Phương trình động lực học:
    tau = M(q)·qddot + C(q, qdot)·qdot + G(q) + F(qdot)

Nguồn công thức: document/Kết quả tính toán M,G,C.docx
và ESP32 on_board_ctl.h/cpp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin, sqrt
from typing import Sequence


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
PRISMATIC_MIN_MM = 350.0
PRISMATIC_MAX_MM = 450.0
PRISMATIC_BASE_MM = 350.0


# ─────────────────────────────────────────────────────────────────────────────
#  Cấu hình khâu (Link Parameters)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LinkInertia3D:
    """Ma trận quán tính 3x3 của khâu quay (quanh trục Z)."""
    Ixx: float = 0.0
    Iyy: float = 0.0
    Izz: float = 0.0
    Ixy: float = 0.0
    Ixz: float = 0.0
    Iyz: float = 0.0

    @classmethod
    def from_scalar(cls, Izz: float, mass: float = 0.0, com_x: float = 0.0, com_y: float = 0.0):
        Ixx = Izz + mass * com_y**2
        Iyy = Izz + mass * com_x**2
        Ixy = -mass * com_x * com_y
        return cls(Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixy=Ixy, Ixz=0.0, Iyz=0.0)


@dataclass
class JointParams:
    """Thông số vật lý của mỗi khâu robot.
    
    Cấu hình:
        j0 (joints[0]): Hip - gear_ratio=100, motor_inertia_big
        j1 (joints[1]): Knee - gear_ratio=50, motor_inertia_small
        j2 (joints[2]): Ankle - gear_ratio=50, motor_inertia_small
    """
    mass: float = 1.0
    length: float = 0.35     # l1, l2 (m) - Variable: updated from VL53L0X
    com_distance: float = 0.1  # lc1, lc2, lc3 (m)
    
    # Motor parameters
    motor_inertia: float = 0.0
    gear_ratio: float = 1.0
    
    # Ma trận quán tính 3x3
    inertia: LinkInertia3D | float = field(default_factory=LinkInertia3D)

    def __post_init__(self):
        if isinstance(self.inertia, (int, float)):
            Izz = float(self.inertia)
            self.inertia = LinkInertia3D(Izz=Izz)
    
    @property
    def reflected_inertia(self) -> float:
        """I_zz reflected = I_zz + m*r^2 + GR^2 * motor_inertia"""
        I_zz = self.inertia.Izz if hasattr(self.inertia, 'Izz') else self.inertia
        return (I_zz 
                + self.mass * self.com_distance**2 
                + self.gear_ratio**2 * self.motor_inertia)


@dataclass
class CTC3Gains:
    kp: tuple[float, float, float]
    kd: tuple[float, float, float]


@dataclass
class CTC3Model:
    """Model vật lý đầy đủ cho 3-DOF R-R-R robot.
    
    Prismatic joints ảnh hưởng đến chiều dài:
        - joints[0].length = l1 (hip link length, m)
        - joints[1].length = l2 (knee link length, m)
        - joints[0].com_distance = lc1 (hip COM, m)
        - joints[1].com_distance = lc2 (knee COM, m)
        - joints[2].com_distance = lc3 (ankle COM, m)
    
    Prismatic position (mm) - từ VL53L0X:
        - prismatic_hip_mm: 350-450 mm
        - prismatic_knee_mm: 350-450 mm
    """
    joints: tuple[JointParams, JointParams, JointParams]
    gravity: float = 9.81
    coulomb_friction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    viscous_friction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_scale: float = 1.0
    # Prismatic positions (mm) from VL53L0X
    prismatic_hip_mm: float = 350.0
    prismatic_knee_mm: float = 350.0
    
    def update_prismatic_lengths(self) -> None:
        """Cập nhật chiều dài khâu từ prismatic position (mm → m).
        
        hip_link_length = hip_prismatic_mm / 1000.0
        knee_link_length = knee_prismatic_mm / 1000.0
        """
        # Clamp to valid range
        hip_mm = max(PRISMATIC_MIN_MM, min(PRISMATIC_MAX_MM, self.prismatic_hip_mm))
        knee_mm = max(PRISMATIC_MIN_MM, min(PRISMATIC_MAX_MM, self.prismatic_knee_mm))
        
        self.prismatic_hip_mm = hip_mm
        self.prismatic_knee_mm = knee_mm
        self.joints[0].length = hip_mm / 1000.0
        self.joints[1].length = knee_mm / 1000.0


def _as_3(seq: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(seq) != 3:
        raise ValueError(f"{name} must have length 3")
    return float(seq[0]), float(seq[1]), float(seq[2])


def _sign(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Mass Matrix M(q) cho R-R-R
# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình R-R-R:
#   q = [q1, q2, q3] where:
#   - q1: góc hip (revolute) - quay quanh Z
#   - q2: góc knee (revolute) - quay quanh Z
#   - q3: góc ankle (revolute) - quay quanh Z
#
# Chiều dài biến (do prismatic joints bên trong):
#   l1 = joints[0].length (hip link)
#   l2 = joints[1].length (knee link)
#   lc1 = joints[0].com_distance (hip COM)
#   lc2 = joints[1].com_distance (knee COM)
#   lc3 = joints[2].com_distance (ankle COM)
#
# Công thức (đã match với ESP32 on_board_ctl.h):
# M[0][0] = I0+I1+I2 + m0*r1² + m1*(l1²+r2²+2*l1*r2*cos(q2))
#           + m2*(l1²+l2²+r3²+2*l1*l2*cos(q2)+2*l1*r3*cos(q2+q3)+2*l2*r3*cos(q3))
# M[0][1] = M[1][0] = I1+I2 + m1*(r2²+l1*r2*cos(q2))
#           + m2*(l2²+r3²+l1*l2*cos(q2)+l1*r3*cos(q2+q3)+l2*r3*cos(q3))
# M[0][2] = M[2][0] = I2 + m2*(r3²+l1*r3*cos(q2+q3)+l2*r3*cos(q3))
# M[1][1] = I1+I2 + m1*r2² + m2*(l2²+r3²+2*l2*r3*cos(q3))
# M[1][2] = M[2][1] = I2 + m2*(r3²+l2*r3*cos(q3))
# M[2][2] = I2 + m2*r3²

def _mass_matrix(model: CTC3Model, q: tuple[float, float, float]) -> list[list[float]]:
    """Mass matrix for R-R-R (3 Revolute Joints)."""
    j0, j1, j2 = model.joints
    q1, q2, q3 = q
    
    # Update lengths from prismatic sensors
    l1 = j0.length   # hip link length (m)
    l2 = j1.length   # knee link length (m)
    lc1 = j0.com_distance  # hip COM
    lc2 = j1.com_distance  # knee COM
    lc3 = j2.com_distance  # ankle COM
    
    m0 = j0.mass  # hip mass
    m1 = j1.mass  # knee mass
    m2 = j2.mass  # ankle mass
    
    # Inertia (reflected to joint)
    I0 = j0.reflected_inertia
    I1 = j1.reflected_inertia
    I2 = j2.reflected_inertia
    
    # Trigonometry
    c2 = cos(q2)
    c3 = cos(q3)
    c23 = cos(q2 + q3)
    
    # M[0][0]
    M00 = (I0 + I1 + I2
           + m0 * lc1**2
           + m1 * (l1**2 + lc2**2 + 2*l1*lc2*c2)
           + m2 * (l1**2 + l2**2 + lc3**2
                   + 2*l1*l2*c2 + 2*l1*lc3*c23 + 2*l2*lc3*c3))
    
    # M[0][1] = M[1][0]
    M01 = (I1 + I2
           + m1 * (lc2**2 + l1*lc2*c2)
           + m2 * (l2**2 + lc3**2
                   + l1*l2*c2 + l1*lc3*c23 + l2*lc3*c3))
    
    # M[0][2] = M[2][0]
    M02 = (I2
           + m2 * (lc3**2 + l1*lc3*c23 + l2*lc3*c3))
    
    # M[1][1]
    M11 = (I1 + I2
           + m1 * lc2**2
           + m2 * (l2**2 + lc3**2 + 2*l2*lc3*c3))
    
    # M[1][2] = M[2][1]
    M12 = (I2
           + m2 * (lc3**2 + l2*lc3*c3))
    
    # M[2][2]
    M22 = I2 + m2 * lc3**2
    
    return [
        [M00, M01, M02],
        [M01, M11, M12],
        [M02, M12, M22],
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Coriolis Matrix C(q, qdot) cho R-R-R
# ─────────────────────────────────────────────────────────────────────────────
# Công thức (đã match với ESP32 on_board_ctl.h):
# C[0][0] = -m1*l1*lc2*dq2*sin(q2) - m2*l1*l2*dq2*sin(q2) 
#           - m2*l1*lc3*(dq2+dq3)*sin(q2+q3) - m2*l2*lc3*dq3*sin(q3)
# C[0][1] = -m1*l1*lc2*(dq1+dq2)*sin(q2) - m2*l1*l2*(dq1+dq2)*sin(q2)
#           - m2*l1*lc3*(dq1+dq2+dq3)*sin(q2+q3) - m2*l2*lc3*dq3*sin(q3)
# C[0][2] = -m2*l1*lc3*(dq1+dq2+dq3)*sin(q2+q3) - m2*l2*lc3*(dq1+dq2+dq3)*sin(q3)
# C[1][0] = m1*l1*lc2*dq1*sin(q2) + m2*l1*l2*dq1*sin(q2) 
#           + m2*l1*lc3*dq1*sin(q2+q3) - m2*l2*lc3*dq3*sin(q3)
# C[1][1] = -m2*l2*lc3*dq3*sin(q3)
# C[1][2] = -m2*l2*lc3*(dq1+dq2+dq3)*sin(q3)
# C[2][0] = m2*l1*lc3*dq1*sin(q2+q3) + m2*l2*lc3*dq2*sin(q3)
# C[2][1] = m2*l1*lc3*(dq1+dq2)*sin(q2+q3) + m2*l2*lc3*dq2*sin(q3)
# C[2][2] = 0

def _coriolis_matrix(model: CTC3Model, q: tuple[float, float, float], q_dot: tuple[float, float, float]) -> list[list[float]]:
    """Coriolis/Centrifugal matrix for R-R-R."""
    j0, j1, j2 = model.joints
    q1, q2, q3 = q
    dq1, dq2, dq3 = q_dot
    
    # Get lengths
    l1 = j0.length
    l2 = j1.length
    lc2 = j1.com_distance
    lc3 = j2.com_distance
    
    m1 = j1.mass
    m2 = j2.mass
    
    # Trigonometry
    s2 = sin(q2)
    s3 = sin(q3)
    s23 = sin(q2 + q3)
    
    # C[0][0]
    C00 = (-m1*l1*lc2*dq2*s2
           - m2*l1*l2*dq2*s2
           - m2*l1*lc3*(dq2+dq3)*s23
           - m2*l2*lc3*dq3*s3)
    
    # C[0][1]
    C01 = (-m1*l1*lc2*(dq1+dq2)*s2
           - m2*l1*l2*(dq1+dq2)*s2
           - m2*l1*lc3*(dq1+dq2+dq3)*s23
           - m2*l2*lc3*dq3*s3)
    
    # C[0][2]
    C02 = (-m2*l1*lc3*(dq1+dq2+dq3)*s23
           - m2*l2*lc3*(dq1+dq2+dq3)*s3)
    
    # C[1][0]
    C10 = (m1*l1*lc2*dq1*s2
           + m2*l1*l2*dq1*s2
           + m2*l1*lc3*dq1*s23
           - m2*l2*lc3*dq3*s3)
    
    # C[1][1]
    C11 = -m2*l2*lc3*dq3*s3
    
    # C[1][2]
    C12 = -m2*l2*lc3*(dq1+dq2+dq3)*s3
    
    # C[2][0]
    C20 = m2*l1*lc3*dq1*s23 + m2*l2*lc3*dq2*s3
    
    # C[2][1]
    C21 = m2*l1*lc3*(dq1+dq2)*s23 + m2*l2*lc3*dq2*s3
    
    # C[2][2] = 0
    C22 = 0.0
    
    return [
        [C00, C01, C02],
        [C10, C11, C12],
        [C20, C21, C22],
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Gravity Vector G(q) cho R-R-R
# ─────────────────────────────────────────────────────────────────────────────
# Công thức (đã match với ESP32 on_board_ctl.h):
# G[0] = g * (m0*lc1*sin(q1) + m1*(l1*sin(q1) + lc2*sin(q1+q2)) 
#            + m2*(l1*sin(q1) + l2*sin(q1+q2) + lc3*sin(q1+q2+q3)))
# G[1] = g * (m1*lc2*sin(q1+q2) + m2*(l2*sin(q1+q2) + lc3*sin(q1+q2+q3)))
# G[2] = g * (m2*lc3*sin(q1+q2+q3))

def _gravity_vector(model: CTC3Model, q: tuple[float, float, float]) -> tuple[float, float, float]:
    """Gravity vector for R-R-R."""
    j0, j1, j2 = model.joints
    q1, q2, q3 = q
    
    g = model.gravity
    m0 = j0.mass
    m1 = j1.mass
    m2 = j2.mass
    
    l1 = j0.length
    l2 = j1.length
    lc1 = j0.com_distance
    lc2 = j1.com_distance
    lc3 = j2.com_distance
    
    # Angles
    s1 = sin(q1)
    s12 = sin(q1 + q2)
    s123 = sin(q1 + q2 + q3)
    
    # G[0] = torque at hip
    G0 = g * (m0*lc1*s1
              + m1*(l1*s1 + lc2*s12)
              + m2*(l1*s1 + l2*s12 + lc3*s123))
    
    # G[1] = torque at knee
    G1 = g * (m1*lc2*s12
              + m2*(l2*s12 + lc3*s123))
    
    # G[2] = torque at ankle
    G2 = g * m2*lc3*s123
    
    return (G0, G1, G2)


# ─────────────────────────────────────────────────────────────────────────────
#  Main CTC Functions
# ─────────────────────────────────────────────────────────────────────────────

def ctc_3dof_components(
    qd: Sequence[float],
    q: Sequence[float],
    qd_dot: Sequence[float],
    q_dot: Sequence[float],
    qd_ddot: Sequence[float],
    gains: CTC3Gains,
    model: CTC3Model,
    startup_t: float = 0.0,
    smooth_startup: bool = True,
    startup_duration: float = 0.8,
) -> dict[str, tuple[float, float, float]]:
    """Tính toán đầy đủ các thành phần CTC cho 3-DOF R-R-R.
    
    Args:
        qd: desired joint positions [q1d, q2d, q3d] (rad)
        q: actual joint positions [q1, q2, q3] (rad)
        qd_dot: desired joint velocities [qd1_dot, qd2_dot, qd3_dot] (rad/s)
        q_dot: actual joint velocities [q1_dot, q2_dot, q3_dot] (rad/s)
        qd_ddot: desired joint accelerations [qd1_ddot, qd2_ddot, qd3_ddot] (rad/s²)
        gains: PID gains (kp, kd)
        model: robot physical model (prismatic lengths auto-updated)
        startup_t: time since startup (s)
        smooth_startup: enable gravity ramp-up at startup
        startup_duration: gravity ramp-up duration (s)
    
    Returns:
        dict với 'tau' = (tau1, tau2, tau3) torque outputs (Nm)
    """
    qd = _as_3(qd, "qd")
    q = _as_3(q, "q")
    qd_dot = _as_3(qd_dot, "qd_dot")
    q_dot = _as_3(q_dot, "q_dot")
    qd_ddot = _as_3(qd_ddot, "qd_ddot")

    # Update prismatic lengths from sensors
    model.update_prismatic_lengths()

    # PD feedback: v = qddot_d + Kp*e + Kd*de
    e = tuple(qd[i] - q[i] for i in range(3))
    de = tuple(qd_dot[i] - q_dot[i] for i in range(3))
    p_term = tuple(gains.kp[i] * e[i] for i in range(3))
    d_term = tuple(gains.kd[i] * de[i] for i in range(3))
    v = tuple(qd_ddot[i] + d_term[i] + p_term[i] for i in range(3))

    # Compute M, C, G
    M = _mass_matrix(model, q)
    C = _coriolis_matrix(model, q, q_dot)
    G = _gravity_vector(model, q)

    # Gravity ramp-up at startup (smooth transition)
    if smooth_startup and startup_duration > 0:
        w = max(0.0, min(1.0, startup_t / startup_duration))
    else:
        w = 1.0

    # Compute gravity at current pose for smoothing
    G_hold = _gravity_vector(model, q)

    # Compute torque: tau = M*v + C*qdot + G
    mv = [0.0, 0.0, 0.0]
    cv = [0.0, 0.0, 0.0]
    tau: list[float] = []
    
    for i in range(3):
        for j in range(3):
            mv[i] += M[i][j] * v[j]
            cv[i] += C[i][j] * q_dot[j]
        
        # Smooth gravity transition at startup
        G_start_i = G_hold[i]
        G_smooth_i = G_start_i * (1.0 - w) + G[i] * w
        
        tau_i = mv[i] + cv[i] + G_smooth_i
        
        # Add friction
        tau_i += model.viscous_friction[i] * q_dot[i]
        tau_i += model.coulomb_friction[i] * _sign(q_dot[i])
        
        tau.append(tau_i * model.torque_scale)

    return {
        "e": e, "de": de, "p_term": p_term, "d_term": d_term, "v": v,
        "mv": tuple(mv), "cv": tuple(cv), "g": G, "g_hold": G_hold, "g_ff": G,
        "startup_w": w,
        "friction": tuple(
            model.viscous_friction[i] * q_dot[i] + model.coulomb_friction[i] * _sign(q_dot[i])
            for i in range(3)
        ),
        "tau": tuple(tau),
        "M": M, "C": C,
        "prismatic_hip_mm": model.prismatic_hip_mm,
        "prismatic_knee_mm": model.prismatic_knee_mm,
        "hip_link_length_m": model.joints[0].length,
        "knee_link_length_m": model.joints[1].length,
    }


def ctc_3dof(
    qd: Sequence[float],
    q: Sequence[float],
    qd_dot: Sequence[float],
    q_dot: Sequence[float],
    qd_ddot: Sequence[float],
    gains: CTC3Gains,
    model: CTC3Model,
    startup_t: float = 0.0,
    smooth_startup: bool = True,
    startup_duration: float = 0.8,
) -> tuple[float, float, float]:
    """Tính torque output cho 3-DOF R-R-R robot.
    
    Returns:
        (tau1, tau2, tau3) - torques in Nm
    """
    return ctc_3dof_components(qd, q, qd_dot, q_dot, qd_ddot, gains, model, startup_t, smooth_startup, startup_duration)["tau"]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper để tạo model từ tham số
# ─────────────────────────────────────────────────────────────────────────────

def create_ctc_model_from_params(
    hip_mass: float, hip_length: float, hip_com: float, hip_inertia: float,
    hip_motor_inertia: float, hip_gear_ratio: float,
    knee_mass: float, knee_length: float, knee_com: float, knee_inertia: float,
    knee_motor_inertia: float, knee_gear_ratio: float,
    ankle_mass: float, ankle_length: float, ankle_com: float, ankle_inertia: float,
    ankle_motor_inertia: float, ankle_gear_ratio: float,
    gravity: float = 9.81,
    coulomb_friction: tuple[float, float, float] = (0.05, 0.0, 0.02),
    viscous_friction: tuple[float, float, float] = (0.002, 0.0, 0.001),
    prismatic_hip_mm: float = 350.0,
    prismatic_knee_mm: float = 350.0,
) -> CTC3Model:
    """Tạo CTC3Model từ các tham số vật lý.
    
    Args:
        hip_*: Tham số khâu hip (q1)
        knee_*: Tham số khâu knee (q2)
        ankle_*: Tham số khâu ankle (q3)
        gravity: Gia tốc trọng trường (m/s²)
        coulomb_friction: Coulomb friction cho từng khớp (Nm)
        viscous_friction: Viscous friction cho từng khớp (Nm·s/rad)
        prismatic_hip_mm: Vị trí prismatic hip (mm, 350-450)
        prismatic_knee_mm: Vị trí prismatic knee (mm, 350-450)
    """
    
    def make_joint(mass, length, com, inertia, motor_inertia, gear_ratio) -> JointParams:
        return JointParams(
            mass=mass, length=length, com_distance=com,
            inertia=LinkInertia3D.from_scalar(inertia, mass, com, 0),
            motor_inertia=motor_inertia, gear_ratio=gear_ratio,
        )
    
    model = CTC3Model(
        joints=(
            make_joint(hip_mass, hip_length, hip_com, hip_inertia, hip_motor_inertia, hip_gear_ratio),
            make_joint(knee_mass, knee_length, knee_com, knee_inertia, knee_motor_inertia, knee_gear_ratio),
            make_joint(ankle_mass, ankle_length, ankle_com, ankle_inertia, ankle_motor_inertia, ankle_gear_ratio),
        ),
        gravity=gravity,
        coulomb_friction=coulomb_friction,
        viscous_friction=viscous_friction,
        prismatic_hip_mm=prismatic_hip_mm,
        prismatic_knee_mm=prismatic_knee_mm,
    )
    
    # Update lengths from prismatic positions
    model.update_prismatic_lengths()
    
    return model
