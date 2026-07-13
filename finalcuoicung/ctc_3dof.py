"""3-DOF R-R-R Computed Torque Control (CTC) - matching Robot_Dynamics_Code.c.

Cấu hình robot:
    Joint 1 (q1): Revolute - Hip xoay quanh trục Z
    Joint 2 (q2): Revolute - Knee xoay quanh trục Z
    Joint 3 (q3): Revolute - Ankle xoay quanh trục Z

Prismatic joints bên trong mỗi khâu ảnh hưởng đến:
    - l1 = joints[0].length: Chiều dài hip link (350-450 mm từ VL53L0X)
    - l2 = joints[1].length: Chiều dài knee link (350-450 mm từ VL53L0X)
    - xc1,yc1 = joints[0].com_x,com_y: COM hip (hip_link_chung_x, hip_link_chung_y)
    - xc2,yc2 = joints[1].com_x,com_y: COM knee (knee_link_chung_x, knee_link_chung_y)
    - xc3,yc3 = joints[2].com_x,com_y: COM ankle (ankle_link_com_distance, 0)

Phương trình động lực học:
    tau = M(q)·qddot + C(q, qdot)·qdot + G(q) + F(qdot)

Nguồn công thức: document/Kết quả tính toán M,G,C.docx
và ESP32 on_board_ctl.h/cpp, Robot_Dynamics_Code.c

Lưu ý: Gravity vector G đã được điều chỉnh để match với C code.
Nếu yc1=yc2=yc3=0 (mặc định), G = g*m*r*cos(θ), không phải sin(θ).
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
    
    2D COM vector [xc, yc] - matching Robot_Dynamics_Code.c
    xc = hip_link_chung_x, yc = hip_link_chung_y (etc.)
    """
    mass: float = 1.0
    length: float = 0.35     # l1, l2 (m) - Variable: updated from VL53L0X
    
    # 2D COM vector (matching C code xc1,yc1, xc2,yc2, xc3,yc3)
    com_x: float = 0.0  # xc - COM distance along link axis
    com_y: float = 0.0  # yc - COM offset perpendicular (tangential)
    
    # Motor parameters
    motor_inertia: float = 0.0
    gear_ratio: float = 1.0
    
    # Inertia (I_zz about COM axis)
    inertia: LinkInertia3D | float = field(default_factory=LinkInertia3D)
    
    @property
    def com_distance(self) -> float:
        """Khoảng cách COM từ joint (sqrt(xc^2 + yc^2))."""
        return sqrt(self.com_x**2 + self.com_y**2)
    
    def __post_init__(self):
        if isinstance(self.inertia, (int, float)):
            self.inertia = float(self.inertia)


@dataclass
class CTC3Gains:
    kp: tuple[float, float, float]
    kd: tuple[float, float, float]


@dataclass
class CTC3Model:
    """Model vật lý đầy đủ cho 3-DOF R-R-R robot (matching Robot_Dynamics_Code.c).

    Joints:
        - joints[0]: Hip (q1) - l1, xc1,yc1, mass
        - joints[1]: Knee (q2) - l2, xc2,yc2, mass
        - joints[2]: Ankle (q3) - l3, xc3,yc3, mass

    Prismatic position (mm) - từ VL53L0X:
        - prismatic_hip_mm: 350-450 mm
        - prismatic_knee_mm: 350-450 mm

    Friction model:
        - viscous_friction[i]: B·qdot (Nm·s/rad)
        - coulomb_friction[i]: T_c·sgn(qdot) (Nm)
        - static_friction[i]: T_s·direction(ep) chỉ khi |qdot| < stribeck_vel_thresh
          (Nm) — còn gọi là Stribeck effect. Quan trọng khi motor đứng yên cần
          vượt ngưỡng ma sát tĩnh để bắt đầu chuyển động.
    """
    joints: tuple[JointParams, JointParams, JointParams]
    gravity: float = 9.81
    static_friction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    coulomb_friction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    viscous_friction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Ngưỡng |qdot| để kích hoạt static friction (rad/s). Mặc định 2 deg/s ≈ 0.0349 rad/s
    # — cùng ngưỡng với ODESC `vel_threshole=2` (deg/s) trong Trajectory_controller.py.
    stribeck_vel_thresh: float = 0.0349
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


def _smooth_sign(x: float, k: float = 50.0) -> float:
    """sign mượt = tanh(k*x) — thay cho `sign()` rời rạc để tránh torque rung
    khi qdot dao động quanh 0 do encoder noise."""
    import math
    return math.tanh(k * x)


# ─────────────────────────────────────────────────────────────────────────────
#  Mass Matrix M(q) cho R-R-R - matching Robot_Dynamics_Code.c
# ─────────────────────────────────────────────────────────────────────────────
# M[0][0]= Izz1+Izz2+Izz3+(L1*L1)*m2+(L1*L1)*m3+(L2*L2)*m3
#          +m1*(xc1²+yc1²)+m2*(xc2²+yc2²)+m3*(xc3²+yc3²)
#          +L1*m3*xc3*cos(q2+q3)*2+L1*L2*m3*cos(q2)*2-L1*m3*yc3*sin(q2+q3)*2
#          +L1*m2*xc2*cos(q2)*2+L2*m3*xc3*cos(q3)*2-L1*m2*yc2*sin(q2)*2-L2*m3*yc3*sin(q3)*2
# M[0][1]= M[1][0]= Izz2+Izz3+(L2*L2)*m3+m2*(xc2²+yc2²)+m3*(xc3²+yc3²)
#          +L1*m3*xc3*cos(q2+q3)+L1*L2*m3*cos(q2)-L1*m3*yc3*sin(q2+q3)
#          +L1*m2*xc2*cos(q2)+L2*m3*xc3*cos(q3)*2-L1*m2*yc2*sin(q2)-L2*m3*yc3*sin(q3)*2
# M[0][2]= M[2][0]= Izz3+m3*(xc3²+yc3²)+L1*m3*xc3*cos(q2+q3)-L1*m3*yc3*sin(q2+q3)
#          +L2*m3*xc3*cos(q3)-L2*m3*yc3*sin(q3)
# M[1][1]= Izz2+Izz3+(L2*L2)*m3+m2*(xc2²+yc2²)+m3*(xc3²+yc3²)
#          +L2*m3*xc3*cos(q3)*2-L2*m3*yc3*sin(q3)*2
# M[1][2]= M[2][1]= Izz3+m3*(xc3²+yc3²)+L2*m3*xc3*cos(q3)-L2*m3*yc3*sin(q3)
# M[2][2]= Izz3+m3*(xc3²+yc3²)

def _mass_matrix(model: CTC3Model, q: tuple[float, float, float]) -> list[list[float]]:
    j0, j1, j2 = model.joints
    q1, q2, q3 = q
    
    l1 = j0.length   # L1
    l2 = j1.length   # L2
    xc1 = j0.com_x   # xc1 = hip_link_chung_x
    yc1 = j0.com_y   # yc1 = hip_link_chung_y
    xc2 = j1.com_x   # xc2 = knee_link_chung_x
    yc2 = j1.com_y   # yc2 = knee_link_chung_y
    xc3 = j2.com_x   # xc3 = ankle_link_com_distance
    yc3 = j2.com_y   # yc3 = 0
    
    # Python variable convention (KHÔNG theo C):
    #   m0 = hip mass (= m1 trong C/Robot_Dynamics_Code.c)
    #   m1 = knee mass (= m2 trong C)
    #   m2 = ankle mass (= m3 trong C)
    # Công thức bên dưới giữ nguyên convention C, dùng aliases m1_c/m2_c/m3_c.
    m0 = j0.mass
    m1 = j1.mass
    m2 = j2.mass
    m1_c, m2_c, m3_c = m0, m1, m2

    Izz1 = j0.inertia if isinstance(j0.inertia, float) else j0.inertia.Izz
    Izz2 = j1.inertia if isinstance(j1.inertia, float) else j1.inertia.Izz
    Izz3 = j2.inertia if isinstance(j2.inertia, float) else j2.inertia.Izz

    # Reflect motor rotor inertia to joint-side
    # tau_motor = I_motor * qddot_motor. Khi phan hoi qua hop so (gear_ratio N),
    # qddot_joint = qddot_motor * N va tau_joint = tau_motor * N. Vay I_motor dong
    # gop vao M(q) o joint-side la I_motor * N^2. Neu bo qua, moment CTC se thieu
    # rat nhieu (hip gear=100: 0.002676 * 100^2 = 26.76 kg.m^2 -> gap ~260 lan
    # link inertia), motor khong du moment quan tinh de bam qddot_d.
    Izz1 += j0.motor_inertia * j0.gear_ratio ** 2
    Izz2 += j1.motor_inertia * j1.gear_ratio ** 2
    Izz3 += j2.motor_inertia * j2.gear_ratio ** 2

    c2 = cos(q2)
    c3 = cos(q3)
    s2 = sin(q2)
    s3 = sin(q3)
    c23 = cos(q2 + q3)
    s23 = sin(q2 + q3)

    # M[0][0]
    M00 = (Izz1 + Izz2 + Izz3
           + l1*l1*m2_c + l1*l1*m3_c + l2*l2*m3_c
           + m0*(xc1*xc1) + m1*(xc2*xc2) + m2*(xc3*xc3)
           + m0*(yc1*yc1) + m1*(yc2*yc2) + m2*(yc3*yc3)
           + l1*m2_c*xc3*c23*2.0 + l1*l2*m3_c*c2*2.0
           - l1*m2_c*yc3*s23*2.0 + l1*m1_c*xc2*c2*2.0
           + l2*m2_c*xc3*c3*2.0 - l1*m1_c*yc2*s2*2.0 - l2*m2_c*yc3*s3*2.0)

    # M[0][1]
    M01 = (Izz2 + Izz3 + l2*l2*m3_c
           + m1*(xc2*xc2) + m2*(xc3*xc3)
           + m1*(yc2*yc2) + m2*(yc3*yc3)
           + l1*m2_c*xc3*c23 + l1*l2*m3_c*c2
           - l1*m2_c*yc3*s23 + l1*m1_c*xc2*c2
           + l2*m2_c*xc3*c3*2.0 - l1*m1_c*yc2*s2 - l2*m2_c*yc3*s3*2.0)

    # M[0][2]
    M02 = (Izz3 + m2*(xc3*xc3) + m2*(yc3*yc3)
           + l1*m2_c*xc3*c23 - l1*m2_c*yc3*s23
           + l2*m2_c*xc3*c3 - l2*m2_c*yc3*s3)

    # M[1][0] = M[0][1]
    M10 = M01

    # M[1][1]
    M11 = (Izz2 + Izz3 + l2*l2*m3_c
           + m1*(xc2*xc2) + m2*(xc3*xc3)
           + m1*(yc2*yc2) + m2*(yc3*yc3)
           + l2*m2_c*xc3*c3*2.0 - l2*m2_c*yc3*s3*2.0)

    # M[1][2]
    M12 = (Izz3 + m2*(xc3*xc3) + m2*(yc3*yc3)
           + l2*m2_c*xc3*c3 - l2*m2_c*yc3*s3)
    
    # M[2][0] = M[0][2]
    M20 = M02
    
    # M[2][1] = M[1][2]
    M21 = M12
    
    # M[2][2]
    M22 = Izz3 + m2*(xc3*xc3 + yc3*yc3)
    
    return [
        [M00, M01, M02],
        [M10, M11, M12],
        [M20, M21, M22],
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Coriolis Matrix C(q, qdot) - matching Robot_Dynamics_Code.c
# ─────────────────────────────────────────────────────────────────────────────
# C[0][0] = -dq3*m3*(L2*xc3*sin(q3)+L1*yc3*cos(q2+q3)+L1*xc3*sin(q2+q3)+L2*yc3*cos(q3))
#            -L1*dq2*(m2*yc2*cos(q2)+m2*xc2*sin(q2)+m3*yc3*cos(q2+q3)+m3*xc3*sin(q2+q3)+L2*m3*sin(q2))
# C[0][1] = -dq3*m3*(...)-L1*dq1*(...)-L1*dq2*(...)
# C[0][2] = -m3*(dq1+dq2+dq3)*(L2*xc3*sin(q3)+L1*yc3*cos(q2+q3)+L1*xc3*sin(q2+q3)+L2*yc3*cos(q3))
# C[1][0] = L1*dq1*(...)-L2*dq3*m3*(yc3*cos(q3)+xc3*sin(q3))
# C[1][1] = -L2*dq3*m3*(yc3*cos(q3)+xc3*sin(q3))
# C[1][2] = -L2*m3*(yc3*cos(q3)+xc3*sin(q3))*(dq1+dq2+dq3)
# C[2][0] = dq1*(L1*m3*yc3*cos(q2+q3)+L1*m3*xc3*sin(q2+q3)+L2*m3*yc3*cos(q3)+L2*m3*xc3*sin(q3))
#            +dq2*(L2*m3*yc3*cos(q3)+L2*m3*xc3*sin(q3))
# C[2][1] = L2*m3*(dq1+dq2)*(yc3*cos(q3)+xc3*sin(q3))
# C[2][2] = 0

def _coriolis_matrix(model: CTC3Model, q: tuple[float, float, float], q_dot: tuple[float, float, float]) -> list[list[float]]:
    j0, j1, j2 = model.joints
    q1, q2, q3 = q
    dq1, dq2, dq3 = q_dot
    
    l1 = j0.length   # L1
    l2 = j1.length   # L2
    xc1 = j0.com_x   # xc1 = hip_link_chung_x
    yc1 = j0.com_y   # yc1 = hip_link_chung_y
    xc2 = j1.com_x   # xc2 = knee_link_chung_x
    yc2 = j1.com_y   # yc2 = knee_link_chung_y
    xc3 = j2.com_x   # xc3 = ankle_com_distance
    yc3 = j2.com_y   # yc3 = 0
    
    m0 = j0.mass  # m1 in C
    m1 = j1.mass  # m2 in C
    m2 = j2.mass  # m3 in C
    
    # Trigonometry
    s2 = sin(q2)
    s3 = sin(q3)
    c2 = cos(q2)
    c3 = cos(q3)
    s23 = sin(q2 + q3)
    c23 = cos(q2 + q3)
    
    # C[0][0]
    C00 = (-dq3*m2*(l2*xc3*s3 + l1*yc3*c23 + l1*xc3*s23 + l2*yc3*c3)
           - l1*dq2*(m1*yc2*c2 + m1*xc2*s2 + m2*yc3*c23 + m2*xc3*s23 + l2*m2*s2))
    
    # C[0][1]
    C01 = (-dq3*m2*(l2*xc3*s3 + l1*yc3*c23 + l1*xc3*s23 + l2*yc3*c3)
           - l1*dq1*(m1*yc2*c2 + m1*xc2*s2 + m2*yc3*c23 + m2*xc3*s23 + l2*m2*s2)
           - l1*dq2*(m1*yc2*c2 + m1*xc2*s2 + m2*yc3*c23 + m2*xc3*s23 + l2*m2*s2))
    
    # C[0][2]
    C02 = -m2*(dq1+dq2+dq3)*(l2*xc3*s3 + l1*yc3*c23 + l1*xc3*s23 + l2*yc3*c3)
    
    # C[1][0]
    C10 = (l1*dq1*(m1*yc2*c2 + m1*xc2*s2 + m2*yc3*c23 + m2*xc3*s23 + l2*m2*s2)
           - l2*dq3*m2*(yc3*c3 + xc3*s3))
    
    # C[1][1]
    C11 = -l2*dq3*m2*(yc3*c3 + xc3*s3)
    
    # C[1][2]
    C12 = -l2*m2*(yc3*c3 + xc3*s3)*(dq1+dq2+dq3)
    
    # C[2][0]
    C20 = (dq1*(l1*m2*yc3*c23 + l1*m2*xc3*s23 + l2*m2*yc3*c3 + l2*m2*xc3*s3)
           + dq2*(l2*m2*yc3*c3 + l2*m2*xc3*s3))
    
    # C[2][1]
    C21 = l2*m2*(dq1+dq2)*(yc3*c3 + xc3*s3)
    
    # C[2][2]
    C22 = 0.0
    
    return [
        [C00, C01, C02],
        [C10, C11, C12],
        [C20, C21, C22],
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Gravity Vector G(q) - matching Robot_Dynamics_Code.c
# ─────────────────────────────────────────────────────────────────────────────
# G[0]= g*m2*(xc2*cos(q1+q2)+L1*cos(q1)-yc2*sin(q1+q2))
#       + g*m3*(L2*cos(q1+q2)+L1*cos(q1)+xc3*cos(q1+q2+q3)-yc3*sin(q1+q2+q3))
#       + g*m1*(xc1*cos(q1)-yc1*sin(q1))
# G[1]= g*m2*(xc2*cos(q1+q2)-yc2*sin(q1+q2))
#       + g*m3*(L2*cos(q1+q2)+xc3*cos(q1+q2+q3)-yc3*sin(q1+q2+q3))
# G[2]= g*m3*xc3*cos(q1+q2+q3) - g*m3*yc3*sin(q1+q2+q3)

def _gravity_vector(model: CTC3Model, q: tuple[float, float, float]) -> tuple[float, float, float]:
    j0, j1, j2 = model.joints
    q1, q2, q3 = q

    g = model.gravity
    # Python convention: m0=hip=m1_C, m1=knee=m2_C, m2=ankle=m3_C.
    # Dùng m1_c/m2_c/m3_c để giữ nguyên công thức từ Robot_Dynamics_Code.c.
    m0 = j0.mass
    m1 = j1.mass
    m2 = j2.mass
    m1_c, m2_c, m3_c = m0, m1, m2

    l1 = j0.length   # L1
    l2 = j1.length   # L2
    xc1 = j0.com_x   # xc1 = hip_link_chung_x
    yc1 = j0.com_y   # yc1 = hip_link_chung_y
    xc2 = j1.com_x   # xc2 = knee_link_chung_x
    yc2 = j1.com_y   # yc2 = knee_link_chung_y
    xc3 = j2.com_x   # xc3 = ankle_com_distance
    yc3 = j2.com_y   # yc3 = 0

    c1 = cos(q1)
    s1 = sin(q1)
    c12 = cos(q1 + q2)
    s12 = sin(q1 + q2)
    c123 = cos(q1 + q2 + q3)
    s123 = sin(q1 + q2 + q3)

    # G[0]
    G0 = (g*m2_c*(xc2*c12 + l1*c1 - yc2*s12)
          + g*m3_c*(l2*c12 + l1*c1 + xc3*c123 - yc3*s123)
          + g*m1_c*(xc1*c1 - yc1*s1))

    # G[1]
    G1 = (g*m2_c*(xc2*c12 - yc2*s12)
          + g*m3_c*(l2*c12 + xc3*c123 - yc3*s123))

    # G[2]
    G2 = g*m3_c*xc3*c123 - g*m3_c*yc3*s123

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

    # Pre-compute ep = q - qd (đúng convention ODESC, dùng cho static friction).
    ep = tuple(q[i] - qd[i] for i in range(3))

    for i in range(3):
        joint = model.joints[i]
        for j in range(3):
            mv[i] += M[i][j] * v[j]
            cv[i] += C[i][j] * q_dot[j]

        # Smooth gravity transition at startup
        G_start_i = G_hold[i]
        G_smooth_i = G_start_i * (1.0 - w) + G[i] * w

        tau_i = mv[i] + cv[i] + G_smooth_i

        # ─── Friction ─────────────────────────────────────────────────────
        # Viscous (B·qdot): nhân gear_ratio² để khớp với ODESC (đo motor-side).
        visc_eff = model.viscous_friction[i] * (joint.gear_ratio ** 2)
        tau_i += visc_eff * q_dot[i]
        # Coulomb (T_c·sgn(qdot)): mượt nhờ tanh() để không rung khi |qdot| ≈ 0.
        tau_i += model.coulomb_friction[i] * _smooth_sign(q_dot[i])
        # Stribeck (ma sát tĩnh): chỉ phát huy khi motor gần đứng yên VÀ có lỗi vị trí.
        # Convention ở đây: ep = q - qd (line ~484, ODESC-style). Vật lý: khi motor "kẹt"
        # tại q ≠ qd, cần thêm feedforward torque cùng chiều control để vượt ngưỡng ma sát tĩnh.
        # → stFricDir = sign(ep) = sign(q - qd).
        #   q < qd (undershoot): ep < 0 → stFricDir = -1 → thêm -T_s (cùng chiều control dương).
        #   q > qd (overshoot):  ep > 0 → stFricDir = +1 → thêm +T_s (cùng chiều control âm).
        # Dùng tanh() (k=80) thay sign() rời rạc để không rung khi |e| ≈ 0.
        # Có thêm ramp theo |qdot| để không bật/tắt đột ngột ở ngưỡng vel.
        v_abs = abs(q_dot[i])
        if v_abs < model.stribeck_vel_thresh and model.static_friction[i] != 0.0:
            ramp = 1.0 - v_abs / model.stribeck_vel_thresh
            stFricDir = _smooth_sign(ep[i], k=80.0)           # sign(q - qd), cùng dấu với control term
            tau_i += model.static_friction[i] * stFricDir * ramp

        tau.append(tau_i / model.torque_scale)

    return {
        "e": e, "de": de, "p_term": p_term, "d_term": d_term, "v": v,
        "mv": tuple(mv), "cv": tuple(cv), "g": G, "g_hold": G_hold, "g_ff": G,
        "startup_w": w,
        "friction": tuple(
            model.viscous_friction[i] * q_dot[i] + model.coulomb_friction[i] * _smooth_sign(q_dot[i])
            for i in range(3)
        ),
        "static_friction_torque": tuple(
            (model.static_friction[i] * _smooth_sign(e[i], k=80.0)
             * max(0.0, 1.0 - abs(q_dot[i]) / model.stribeck_vel_thresh)
             if abs(q_dot[i]) < model.stribeck_vel_thresh and model.static_friction[i] != 0.0
             else 0.0)
            for i in range(3)
        ),
        "coulomb_friction_torque": tuple(
            model.coulomb_friction[i] * _smooth_sign(q_dot[i]) for i in range(3)
        ),
        "viscous_friction_torque": tuple(
            model.viscous_friction[i] * q_dot[i] for i in range(3)
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


def ctc_scalar_3dof_components(
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
    """CTC đơn khớp (decoupled) - giống ODESC Trajectory_controller.dynamic_calculation().

    Mỗi khớp tính độc lập với:
        I_i = Izz_i + m_i * com_i² + gear_i² * motor_inertia_i   (scalar)
        G_i = g * m_i * r_i * cos(q_i)                             (không có coupling giữa các khâu)
        C_i = 0                                                    (bỏ Coriolis coupling)

    Công thức pole-cancellation (giống ODESC):
        ep_i  = q_i - qd_i          (convention của ODESC: positive khi overshoot)
        ev_i  = qdot_i - qdot_d_i
        v_i   = qddot_d_i - Kp_i*ep_i - Kd_i*ev_i   (drive error → 0)
        tau_i = I_i * v_i + G_i + F_i

    Ưu điểm so với full M(q)·C(q,qdot):
        - Không bị ảnh hưởng bởi sai số của m_trừ mass mở rộng (chain link mass)
        - Chỉ cần đo đúng I + COM của riêng khâu đó → tune dễ
        - Match với cách ODESC chạy → Kp/Kd transfer được giữa 2 phiên bản

    Nhược điểm:
        - Khi robot coupling mạnh (M1 gập + M2 xoay), torque sai số sẽ lớn
        - Nên cần Kp cao hơn để bù sai số model

    Args:
        Same as ctc_3dof_components.

    Returns:
        dict với 'tau' = (tau1, tau2, tau3) torque outputs (Nm),
        'g' = (G1, G2, G3), 'p_term' = (P1, P2, P3), 'd_term' = (D1, D2, D3),
        'startup_w' = ramp weight, 'm_i' = (I1, I2, I3) scalar inertia.
    """
    qd = _as_3(qd, "qd")
    q = _as_3(q, "q")
    qd_dot = _as_3(qd_dot, "qd_dot")
    q_dot = _as_3(q_dot, "q_dot")
    qd_ddot = _as_3(qd_ddot, "qd_ddot")

    model.update_prismatic_lengths()

    # PD pole-cancellation: ep = q - qd (convention ODESC)
    #   ep > 0 khi overshoot → ta cần torque NGƯỢC chiều → -Kp*ep là giảm.
    ep = tuple(q[i] - qd[i] for i in range(3))
    de = tuple(q_dot[i] - qd_dot[i] for i in range(3))

    # Gravity ramp-up
    if smooth_startup and startup_duration > 0:
        w = max(0.0, min(1.0, startup_t / startup_duration))
    else:
        w = 1.0

    tau: list[float] = []
    g_list: list[float] = []
    p_list: list[float] = []
    d_list: list[float] = []
    mi_list: list[float] = []

    for i in range(3):
        joint = model.joints[i]
        g = model.gravity

        # ── Scalar inertia (giống ODESC: const_inertia + motor_inertia * gear²) ──
        # joint.inertia có thể là float hoặc LinkInertia3D; lấy Izz hoặc giá trị gốc.
        if isinstance(joint.inertia, LinkInertia3D):
            Izz = joint.inertia.Izz
        else:
            Izz = float(joint.inertia)
        I_i = Izz + joint.mass * (joint.com_distance ** 2) + (joint.gear_ratio ** 2) * joint.motor_inertia
        mi_list.append(I_i)

        # ── Gravity scalar (no coupling): G = g * m * r * cos(q) ──
        # Trục tham chiếu q_i đã là góc tuyệt đối (CTC mới cộng model_home_deg ở controller)
        G_i = g * joint.mass * joint.com_distance * cos(q[i])

        # ── PD pole-cancellation ──
        p_i = gains.kp[i] * ep[i]    # positive khi overshoot
        d_i = gains.kd[i] * de[i]
        v_i = qd_ddot[i] - p_i - d_i   # drive error → 0

        # ── Gravity ramp (smooth startup) ──
        G_smooth = G_i * w

        # ── Friction (giống ODESC với stFricDir = sign(ep) đẩy về qd) ──
        # Viscous + Coulomb + Stribeck
        tau_i = I_i * v_i + G_smooth

        # Viscous (B·qdot) — nhân gear_ratio² vì friction được đo ở motor-side (joint-side nhỏ hơn gear² lần)
        # ODESC: visc_friction = 0.00276 * gear_ratio**2 (đã nhân sẵn trong Trajectory_controller.py:74)
        visc_eff = model.viscous_friction[i] * (joint.gear_ratio ** 2)
        tau_i += visc_eff * q_dot[i]
        # Coulomb (T_c·sgn(qdot)) - dùng tanh() mượt
        tau_i += model.coulomb_friction[i] * _smooth_sign(q_dot[i])
        # Stribeck: motor "kẹt" tại q ≠ qd, thêm torque đẩy về qd
        # Convention ODESC: stFricDir = sign(ep) = sign(q - qd)
        v_abs = abs(q_dot[i])
        if v_abs < model.stribeck_vel_thresh and model.static_friction[i] != 0.0:
            ramp = 1.0 - v_abs / model.stribeck_vel_thresh
            stFricDir = _smooth_sign(ep[i], k=80.0)  # sign(q - qd), đẩy về qd
            tau_i += model.static_friction[i] * stFricDir * ramp

        tau.append(tau_i / model.torque_scale)
        g_list.append(G_smooth)
        p_list.append(-p_i)   # log dưới dạng contribution (âm khi overshoot)
        d_list.append(-d_i)

    return {
        "tau": tuple(tau),
        "g": tuple(g_list),
        "p_term": tuple(p_list),
        "d_term": tuple(d_list),
        "startup_w": w,
        "m_i": tuple(mi_list),
    }


def ctc_scalar_3dof(
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
    """Tính torque output cho CTC đơn khớp - same interface as ctc_3dof()."""
    return ctc_scalar_3dof_components(qd, q, qd_dot, q_dot, qd_ddot, gains, model, startup_t, smooth_startup, startup_duration)["tau"]


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
    static_friction: tuple[float, float, float] = (0.092, 0.092, 0.092),
    coulomb_friction: tuple[float, float, float] = (0.05, 0.05, 0.05),
    viscous_friction: tuple[float, float, float] = (0.00276, 0.00276, 0.00276),
    prismatic_hip_mm: float = 350.0,
    prismatic_knee_mm: float = 350.0,
) -> CTC3Model:
    """Tạo CTC3Model từ các tham số vật lý (matching Robot_Dynamics_Code.c).

    Args:
        hip_*: Tham số khâu hip (q1)
        knee_*: Tham số khâu knee (q2)
        ankle_*: Tham số khâu ankle (q3)
        *_com: COM distance = com_x (xc), com_y = 0
        gravity: Gia tốc trọng trường (m/s²)
        static_friction: Ma sát tĩnh (Stribeck) cho từng khớp (Nm).
            Chỉ phát huy khi |qdot| < stribeck_vel_thresh (rad/s). Direction
            = sign(q - qd) → đẩy về setpoint khi motor "kẹt". Default 0.05 Nm
            cho hip/ankle (khớp lớn), 0 cho knee (nếu không đo).
        coulomb_friction: Coulomb friction cho từng khớp (Nm)
        viscous_friction: Viscous friction cho từng khớp (Nm·s/rad)
        prismatic_hip_mm: Vị trí prismatic hip (mm, 350-450)
        prismatic_knee_mm: Vị trí prismatic knee (mm, 350-450)
    """
    
    def make_joint(mass, length, com, inertia, motor_inertia, gear_ratio) -> JointParams:
        return JointParams(
            mass=mass, length=length, com_x=com, com_y=0.0,
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
        static_friction=static_friction,
        coulomb_friction=coulomb_friction,
        viscous_friction=viscous_friction,
        prismatic_hip_mm=prismatic_hip_mm,
        prismatic_knee_mm=prismatic_knee_mm,
    )
    
    # Update lengths from prismatic positions
    model.update_prismatic_lengths()
    
    return model
