import threading
import time
import math
import serial
import serial.tools.list_ports
from collections import deque

import numpy as np

from kinematic import get_acc_jerk
from trajectory import SplineTrajectory, QuinticTrajectory
from ctc_3dof import CTC3Gains, CTC3Model, JointParams, ctc_3dof

# ── Constants (khớp với các controller khác trong project) ──────────────────
AXIS_STATE_IDLE               = 1
AXIS_STATE_CLOSED_LOOP_CONTROL = 8
CLOSED_LOOP_CONTROL           = AXIS_STATE_CLOSED_LOOP_CONTROL
IDLE                          = AXIS_STATE_IDLE

gear_ratio_small = 50.0  
gear_ratio_big = 100.0
DEG2RAD    = math.pi / 180
g          = 9.81

# Device mapping (physical ODrive serial IDs -> logical dev labels)
# dev0: 3489347A3034
# dev1: 3288365C3433
# dev2: 337535753034
DEVICE_ID_MAP = {
    "3489347A3034": "dev0",
    "3288365C3433": "dev1",
    "337535753034": "dev2",
}


# ── Helper ──────────────────────────────────────────────────────────────────
def list_serial_ports():
    """Trả danh sách các COM port khả dụng."""
    return [p.device for p in serial.tools.list_ports.comports()]


# ── Main Controller Class ────────────────────────────────────────────────────
class TWAIController(threading.Thread):

    def __init__(
        self,
        serial_port: str = "COM21",
        baudrate: int = 115200,
    ):
        super().__init__(daemon=True)

        # ── Serial config ────────────────────────────────────────────────
        self.serial_port = serial_port
        self.baudrate    = baudrate
        self.ser: serial.Serial | None = None

        # ── State flags (GUI reads these) ────────────────────────────────
        self.connected          = False
        self.closed_loop_control = False
        self.isOffset           = False
        self.error              = False
        self.esp32_ready        = False   # True sau khi nhận "READY:"
        self.pending_closed_loop = False  # User pressed Enable Torque before ESP32 READY
        self.status_message     = "Chưa kết nối"

        # ── Threading primitives ─────────────────────────────────────────
        self.data_lock   = threading.Lock()
        self._stop_event  = threading.Event()
        self._estop_event = threading.Event()

        # ── Physics / Kinematic (giống các controller cũ) ────────────────
        self.start_pos       = 0.0        # degrees – vị trí gốc khi offset
        # ── Offset (revolutions, raw ODrive value tại thời điểm set_offset) ─
        self.offset_rev = [0.0, 0.0, 0.0]      # offset cho motor 0, 1 và 2

        # ── Position state (degrees) ─────────────────────────────────────
        self.pos  = [0.0, 0.0, 0.0]           # vị trí hiện tại (degrees)
        self.pos_set = [0.0, 0.0, 0.0]        # setpoint (degrees)
        self.vel = [0.0, 0.0, 0.0]            # current velocity (deg/s)
        self.vel_set = [0.0, 0.0, 0.0]        # feedforward velocity (deg/s)
        self._last_pos_set = [0.0, 0.0, 0.0]
        self._last_set_ts = time.perf_counter()
        self._setpoint_dirty = False
        self.motion_armed = False
        self._motion_hold = False
        self._armed_last_pos = [0.0, 0.0, 0.0]
        self._armed_last_time = 0.0
        self._home_pending = False
        self.target_tolerance_deg = 1.0
        self.target_tolerance_vel = 2.0
        self._last_motion_targets = [0.0, 0.0, 0.0]

        # ── CTC plant + spline (cùng cấu trúc trajectory_controller.ODriveThread) ──
        self.trajectory_mode = "spline"
        self.traj      = [SplineTrajectory(), SplineTrajectory(), SplineTrajectory()]
        self.acc_set   = [0.0, 0.0, 0.0]
        self.max_vel   = 5.0  # °/s — giới hạn cho param_calc của spline (GUI có thể mở rộng sau)
        self._motion_t0           = -math.inf
        self._motion_time_active  = False
        self._was_motion_armed    = False
        self.tor_coef  = 0.708282
        self.gear_ratio = gear_ratio_small
        # Per-link physical parameters for the 3-DOF chain
        self.hip_link_mass = 2.4582
        self.knee_link_mass = 2.470
        self.ankle_link_mass = 0.8
        self.hip_link_length = 0.35
        self.knee_link_length = 0.35
        self.ankle_link_length = 0.07
        self.hip_link_com_distance = 0.267
        self.knee_link_com_distance = 0.2694
        self.ankle_link_com_distance = 0.0664
        self.hip_link_inertia = 0.03747833883 + 0.00007289375
        self.knee_link_inertia = 0.03747830028 + 0.00017605071
        self.ankle_link_inertia = 0.00453198525 + 0.00017605071
        self.small_motor_inertia = 0.000643
        self.big_motor_inertia = 0.002676
        self.gear_ratios = (gear_ratio_big, gear_ratio_small, gear_ratio_small)
        # Backward-compatible aliases (legacy code paths)
        self.hip_mass = self.hip_link_mass
        self.knee_mass = self.knee_link_mass
        self.ankle_mass = self.ankle_link_mass
        self.hip_distance = self.hip_link_com_distance
        self.knee_distance = self.knee_link_com_distance
        self.ankle_distance = self.ankle_link_com_distance
        self.gear_ratio = gear_ratio_small
        self.ext_load = 0.0
        self.hanger_mass = 0.0
        self.hanger_distance = 0.0
        self.coul_friction = 0.05  
        self.visc_friction = 0.00276 
        self._recalc_plant_mass_inertia()

        # ── Vận tốc từ FB: dùng perf_counter + LP để tránh dt≈0 khi nhiều FB trong một lần đọc Serial
        self._fb_prev_pc: float | None = None
        self._fb_prev_p = [0.0, 0.0, 0.0]
        self._fb_have_prev = False
        self._vel_lp_alpha = 0.35
        self._vel_max_deg_s = 800.0

        # ── Raw position từ ESP32 (revolutions) ──────────────────────────
        self._raw_pos = [0.0, 0.0, 0.0]  
        self.torque_set = [0.0, 0.0, 0.0]
        self.use_torque_commands = True

        # ── Control params (dùng cho GUI control panel) ──────────────────
        self.Kp_axes = [3.0, 3.0, 3.0]
        self.Kd_axes = [1.0, 1.0, 1.0]
        self.Kp3 = tuple(self.Kp_axes)
        self.Kd3 = tuple(self.Kd_axes)
        self.ctrl_bandwidth = 2000
        self.enc_bandwidth  = 50
        self.max_torque     = 0.3  # torque limit used for saturation in bridge mode
        self.use_torque_mode = True
        self.friction_only_mode = False
        self.friction_only_axis: int | None = None
        self.friction_only_torque = 0.0
        self.locked_axes = [False, False, False]
        self.window_size    = 25
        self.poly_order     = 2
        self.velFilBuf   = [deque(maxlen=self.window_size), deque(maxlen=self.window_size), deque(maxlen=self.window_size)]
        self.timeFilBuf  = [deque(maxlen=self.window_size), deque(maxlen=self.window_size), deque(maxlen=self.window_size)]

        # ── Data buffer (GUI reads for plotting) ─────────────────────────
        # Tuple: (timestamp, pos0_deg, pos1_deg, pos2_deg, pos0_set_deg, pos1_set_deg, pos2_set_deg)
        self.data: deque = deque(maxlen=800)

        # ── Serial receive line buffer ───────────────────────────────────
        self._line_buf = ""
        self._last_status_poll_ts = 0.0
        self._last_torque_log_ts = 0.0

        # Giới hạn tốc độ đổi mô-men (Nm mỗi bước ~10ms) để tránh flip ±max_torque quá nhanh → lắc
        self._tau_out = [0.0, 0.0, 0.0]
        self._tau_slew_frac_cap = 0.22  # tối đa ~22% max_torque mỗi bước vòng lặp
        self.motor_efficiency = (0.90, 0.90, 0.90)
        self.model3 = CTC3Model(
            joints=(
                JointParams(mass=self.hip_link_mass, length=self.hip_link_length, com_distance=self.hip_link_com_distance, inertia=self.hip_link_inertia, motor_inertia=self.big_motor_inertia, gear_ratio=gear_ratio_big),
                JointParams(mass=self.knee_link_mass, length=self.knee_link_length, com_distance=self.knee_link_com_distance, inertia=self.knee_link_inertia, motor_inertia=self.small_motor_inertia, gear_ratio=gear_ratio_small),
                JointParams(mass=self.ankle_link_mass, length=self.ankle_link_length, com_distance=self.ankle_link_com_distance, inertia=self.ankle_link_inertia, motor_inertia=self.small_motor_inertia, gear_ratio=gear_ratio_small),
            ),
            gravity=g,
            coulomb_friction=(self.coul_friction, self.coul_friction, self.coul_friction),
            viscous_friction=(self.visc_friction, self.visc_friction, self.visc_friction),
            torque_scale=self.tor_coef,
        )

    # ════════════════════════════════════════════════════════════════════════
    # Connection
    # ════════════════════════════════════════════════════════════════════════

    def connect(self):
        """Mở cổng Serial. Gọi từ run() hoặc từ GUI thread."""
        try:
            print(f"[TWAI] Kết nối tới {self.serial_port} @ {self.baudrate}...")
            self.ser = serial.Serial(
                self.serial_port,
                self.baudrate,
                timeout=0.02
            )
            self.connected     = True
            self.error         = False
            self.esp32_ready   = False
            self.status_message = f"Đã kết nối {self.serial_port}, đang chờ ESP32..."
            print(f"[TWAI] Kết nối Serial thành công.")
        except Exception as e:
            self.connected      = False
            self.error          = True
            self.status_message = f"Lỗi kết nối: {e}"
            print(f"[TWAI] Lỗi Serial: {e}")

    def disconnect(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.connected = False

    # ════════════════════════════════════════════════════════════════════════
    # State machine (GUI compatibility)
    # ════════════════════════════════════════════════════════════════════════

    def get_state(self):
        if not self.connected:
            return None
        return CLOSED_LOOP_CONTROL if self.closed_loop_control else IDLE

    def _send_simple_cmd(self, cmd: str):
        if not self.ser or not self.ser.is_open:
            return
        try:
            self.ser.write((cmd + "\n").encode("ascii"))
        except Exception as e:
            print(f"[TWAI] Lỗi ghi Serial ({cmd}): {e}")
            self.connected = False

    def enter_closed_loop(self):
        """Kích hoạt chế độ torque/position nhưng chưa chạy motion."""
        if not self.esp32_ready:
            self.pending_closed_loop = True
            self.status_message = "Đang chờ ESP32 READY để vào Closed Loop"
            print("[TWAI] ESP32 chưa READY, sẽ tự vào Closed Loop khi READY.")
            return
        self.pending_closed_loop = False
        self.motion_armed = False
        self._setpoint_dirty = False
        self._send_simple_cmd("CLEAR")
        self._send_simple_cmd("TORQUE" if self.use_torque_mode else "POSITION")
        self._send_simple_cmd("CLOSE")
        self.closed_loop_control = True
        self._reset_torque_slew()
        self.status_message = "Closed Loop sẵn sàng"
        print("[TWAI] Đã vào CLOSED_LOOP_CONTROL.")

    def return_IDLE(self):
        self.pending_closed_loop = False
        self._send_simple_cmd("IDLE")
        self.closed_loop_control = False
        self.motion_armed = False
        self._was_motion_armed = False
        self._motion_time_active = False
        self._setpoint_dirty = False
        self._reset_torque_slew()
        self.status_message = "IDLE"
        print("[TWAI] Trở về IDLE.")

    def is_controlable(self):
        return (
            self.connected
            and self.esp32_ready
            and self.closed_loop_control
            and self.isOffset
            and not self._estop_event.is_set()
        )

    def emergency_stop(self):
        self.pending_closed_loop = False
        self._estop_event.set()
        self._reset_torque_slew()
        self.motion_armed = False
        self._was_motion_armed = False
        self._motion_time_active = False
        self._send_simple_cmd("IDLE")
        self.status_message = "ESTOP!"
        print("[TWAI] EMERGENCY STOP!")

    def reset(self):
        self._estop_event.clear()
        self.pending_closed_loop = False
        self.isOffset = False
        self.motion_armed = False
        self._motion_hold = False
        self._home_pending = False
        self._setpoint_dirty = False
        self._motion_time_active = False
        self._motion_t0 = -math.inf
        self._was_motion_armed = False
        for traj in self.traj:
            traj.reset()
        with self.data_lock:
            self._fb_have_prev = False
            self._fb_prev_pc = None
            self.vel[0] = 0.0
            self.vel[1] = 0.0
            self.acc_set[0] = 0.0
            self.acc_set[1] = 0.0
            self._last_motion_targets[0] = 0.0
            self._last_motion_targets[1] = 0.0
            self._clear_vel_fit_buffers_locked()
            self._reset_torque_slew()
        self.return_IDLE()
        self._send_simple_cmd("CLEAR")
        self.status_message = "Reset xong"
        print("[TWAI] Reset.")

    def stop(self):
        self._stop_event.set()
        self.disconnect()

    # ════════════════════════════════════════════════════════════════════════
    # Offset (lấy vị trí hiện tại làm gốc)
    # ════════════════════════════════════════════════════════════════════════

    def set_offset(self):
        """
        Lưu raw revolutions hiện tại làm offset tham chiếu, giống
        trajectory_controller.py dùng axis.encoder.pos_estimate.
        """
        with self.data_lock:
            self.offset_rev[0] = self._raw_pos[0]
            self.offset_rev[1] = self._raw_pos[1]
            self.offset_rev[2] = self._raw_pos[2]
            self.isOffset = True
            self._fb_have_prev = False
            self._fb_prev_pc = None
            self.vel[0] = self.vel[1] = 0.0
            self._tau_out[0] = self._tau_out[1] = 0.0
            self._clear_vel_fit_buffers_locked()
        print(f"[TWAI] Offset set: motor0={self.offset_rev[0]:.4f} rev, "
              f"motor1={self.offset_rev[1]:.4f} rev")

    # ════════════════════════════════════════════════════════════════════════
    # Unit conversion helpers
    # ════════════════════════════════════════════════════════════════════════

    def _rev_to_deg(self, rev: float, motor_id: int) -> float:
        """ODrive raw rev → degrees (với offset và start_pos)."""
        gear = self.gear_ratios[motor_id] if hasattr(self, "gear_ratios") else self.gear_ratio
        return (rev - self.offset_rev[motor_id]) * 360.0 / gear + self.start_pos

    def _deg_to_rev(self, deg: float, motor_id: int) -> float:
        """Degrees → ODrive raw rev (ngược lại với _rev_to_deg)."""
        gear = self.gear_ratios[motor_id] if hasattr(self, "gear_ratios") else self.gear_ratio
        return (deg - self.start_pos) * gear / 360.0 + self.offset_rev[motor_id]

    def _clear_vel_fit_buffers_locked(self):
        for buf in self.velFilBuf:
            buf.clear()
        for buf in self.timeFilBuf:
            buf.clear()

    def _update_vel_estimates_locked(self, t_wall: float, v0_inst: float, v1_inst: float, v2_inst: float) -> None:
        """
        Nội suy cửa sổ trượt đa thức lên quỹ đạo sai phân v (deg/s) — cùng hàm get_acc_jerk
        như trajectory_controller; TWAI không có encoder vel_estimate nên đây được dùng
        làm đầu vào v thực cho số hạng Kd / ma sát trong CTC.

        trajectory_controller hiện tính nhưng không gán vel_filtered vào self.vel; TWAI chủ động
        gán vào self.vel khi cửa sổ đã đầy.

        kinematic.get_acc_jerk yêu cầu window_size lẻ.
        """
        for i, v_inst in enumerate((v0_inst, v1_inst, v2_inst)):
            self.velFilBuf[i].append(v_inst)
            self.timeFilBuf[i].append(t_wall)
            if len(self.velFilBuf[i]) == self.window_size:
                try:
                    vf, _, _ = get_acc_jerk(
                        np.asarray(self.timeFilBuf[i], dtype=float),
                        np.asarray(self.velFilBuf[i], dtype=float),
                        self.window_size,
                        self.poly_order,
                    )
                    lim = self._vel_max_deg_s
                    self.vel[i] = max(min(float(vf), lim), -lim)
                except ValueError:
                    a = self._vel_lp_alpha
                    self.vel[i] = a * v_inst + (1.0 - a) * self.vel[i]
            else:
                a = self._vel_lp_alpha
                self.vel[i] = a * v_inst + (1.0 - a) * self.vel[i]

    def _recalc_plant_mass_inertia(self):
        """Maintain legacy lumped parameters for compatibility with GUI/test code."""
        self.link_mass = self.hip_mass + self.knee_mass + self.ankle_mass
        self.center_distance = (
            self.hip_mass * self.hip_distance
            + self.knee_mass * self.knee_distance
            + self.ankle_mass * self.ankle_distance
        ) / self.link_mass
        self.const_inertia = (
            self.hip_link_inertia
            + self.knee_link_inertia
            + self.ankle_link_inertia
            + self.hip_mass * (self.hip_distance ** 2)
            + self.knee_mass * (self.knee_distance ** 2)
            + self.ankle_mass * (self.ankle_distance ** 2)
        )
        self.m = self.link_mass + self.hanger_mass + self.ext_load
        self.lc = (self.center_distance * self.link_mass + self.hanger_distance * (self.hanger_mass + self.ext_load)) / max(self.m, 1e-9)
        self.Ic = self.const_inertia + (self.hanger_mass + self.ext_load) * (self.hanger_distance ** 2)

    def _make_trajectory(self):
        if self.trajectory_mode == "quintic":
            return QuinticTrajectory()
        return SplineTrajectory()

    def _set_target_state(self, motor_id: int, target_deg: float):
        """Tính quỹ đạo từ vị trí hiện tại tới target (chỉ gọi trong data_lock)."""
        self.traj[motor_id] = self._make_trajectory()
        self.traj[motor_id].param_calc(self.pos[motor_id], target_deg, self.max_vel)

    def _motion_clock_start(self):
        self._motion_t0 = time.time()
        self._motion_time_active = True

    def _refresh_traj_refs_locked(self, now: float):
        """Cập nhật pos_set / vel_set / acc_set theo thời gian đã trôi của spline."""
        if self._motion_time_active:
            t_prog = max(now - self._motion_t0, 0.0)
        else:
            t_prog = 0.0
        for i in (0, 1, 2):
            p_des, v_des, a_des = self.traj[i].desired_state(t_prog)
            self.pos_set[i], self.vel_set[i], self.acc_set[i] = p_des, v_des, a_des
            if getattr(self, "debug_sign_trace", False):
                tr = self.traj[i]
                print(f"[TWAI][TRACE] desired_state joint={i} traj={type(tr).__name__} t={t_prog:.6f} start={tr.start_p:.6f} end={tr.end_p:.6f} pos={p_des:.6f} vel={v_des:.6f} acc={a_des:.6f}")

    def _dynamic_calculation_locked(self, active_axis: int | None = None):
        """Compute 3-DOF torque commands using the M+C+G CTC model."""
        gains = CTC3Gains(kp=self.Kp3, kd=self.Kd3)
        self.model3.torque_scale = self.tor_coef
        q = [self.pos[i] * DEG2RAD for i in range(3)]
        qd = [self.pos_set[i] * DEG2RAD for i in range(3)]
        qdot = [self.vel[i] * DEG2RAD for i in range(3)]
        qdot_d = [self.vel_set[i] * DEG2RAD for i in range(3)]
        qddot_d = [self.acc_set[i] * DEG2RAD for i in range(3)]
        if getattr(self, "debug_sign_trace", False):
            print(f"[TWAI][TRACE] dynamic q=({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}) qd=({qd[0]:.6f}, {qd[1]:.6f}, {qd[2]:.6f}) qdot=({qdot[0]:.6f}, {qdot[1]:.6f}, {qdot[2]:.6f}) qdot_d=({qdot_d[0]:.6f}, {qdot_d[1]:.6f}, {qdot_d[2]:.6f}) qddot_d=({qddot_d[0]:.6f}, {qddot_d[1]:.6f}, {qddot_d[2]:.6f})")
        tau = ctc_3dof(qd, q, qdot_d, qdot, qddot_d, gains, self.model3)
        self._last_tau_raw = tuple(tau)
        if getattr(self, "debug_sign_trace", False):
            print(f"[TWAI][TRACE] dynamic tau_raw=({tau[0]:.6f}, {tau[1]:.6f}, {tau[2]:.6f}) active_axis={active_axis}")
            print(
                f"[TWAI][TRACE] joint_to_motor_torque mapped=({self._joint_to_motor_torque(0, tau[0]):.6f}, {self._joint_to_motor_torque(1, tau[1]):.6f}, {self._joint_to_motor_torque(2, tau[2]):.6f}) "
                f"gear=({self.gear_ratios[0]:.1f}, {self.gear_ratios[1]:.1f}, {self.gear_ratios[2]:.1f}) eff=({self.motor_efficiency[0]:.2f}, {self.motor_efficiency[1]:.2f}, {self.motor_efficiency[2]:.2f})"
            )
        for i in range(3):
            if active_axis is None or i == active_axis:
                self.torque_set[i] = float(tau[i])
            else:
                self.torque_set[i] = 0.0
        if getattr(self, "debug_sign_trace", False):
            print(f"[TWAI][TRACE] dynamic joint_torque_set=({self.torque_set[0]:.6f}, {self.torque_set[1]:.6f}, {self.torque_set[2]:.6f})")

    def _reset_torque_slew(self):
        self._tau_out[0] = 0.0
        self._tau_out[1] = 0.0
        if len(self._tau_out) > 2:
            self._tau_out[2] = 0.0

    def _joint_to_motor_torque(self, axis: int, tau_joint: float) -> float:
        gear = self.gear_ratios[axis] if hasattr(self, "gear_ratios") else self.gear_ratio
        eff = self.motor_efficiency[axis] if hasattr(self, "motor_efficiency") else 1.0
        eff = max(float(eff), 1e-6)
        return float(tau_joint) / (float(gear) * eff)

    def _motor_to_joint_torque(self, axis: int, tau_motor: float) -> float:
        gear = self.gear_ratios[axis] if hasattr(self, "gear_ratios") else self.gear_ratio
        eff = self.motor_efficiency[axis] if hasattr(self, "motor_efficiency") else 1.0
        return float(tau_motor) * float(gear) * float(eff)

    def _slew_limited_torque(self, axis: int, tau_des: float) -> float:
        prev = self._tau_out[axis]
        max_step = max(self.max_torque * self._tau_slew_frac_cap, 1e-6)
        d = max(min(tau_des - prev, max_step), -max_step)
        out = max(min(prev + d, self.max_torque), -self.max_torque)
        self._tau_out[axis] = out
        return out

    # ════════════════════════════════════════════════════════════════════════
    # Serial Protocol
    # ════════════════════════════════════════════════════════════════════════

    def _send_position(self, motor_id: int, pos_rev: float, vel_rev: float = 0.0):
        """Gửi lệnh setPosition cho motor (đơn vị: revolutions, rev/s)."""
        if not self.ser or not self.ser.is_open:
            return
        cmd = f"P{motor_id}:{pos_rev:.6f},{vel_rev:.6f}\n"
        try:
            self.ser.write(cmd.encode("ascii"))
        except Exception as e:
            print(f"[TWAI] Lỗi ghi Serial: {e}")
            self.connected = False

    def _send_torque(self, motor_id: int, torque_nm: float):
        """Gửi torque command tới ESP32 bridge (đơn vị: Nm)."""
        if not self.ser or not self.ser.is_open:
            return
        torque_nm = max(min(float(torque_nm), self.max_torque), -self.max_torque)
        cmd = f"T{motor_id}:{torque_nm:.6f}\n"
        try:
            self.ser.write(cmd.encode("ascii"))
        except Exception as e:
            print(f"[TWAI] Lỗi ghi Serial torque: {e}")
            self.connected = False

    def _process_serial(self):
        """Đọc và parse các dòng từ ESP32."""
        if not self.ser or not self.ser.is_open:
            return
        try:
            waiting = self.ser.in_waiting
            if waiting > 0:
                raw = self.ser.read(waiting).decode("ascii", errors="replace")
                self._line_buf += raw

                while "\n" in self._line_buf:
                    line, self._line_buf = self._line_buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._parse_line(line)
        except Exception as e:
            print(f"[TWAI] Lỗi đọc Serial: {e}")
            self.connected = False

    def _parse_line(self, line: str):
        """Xử lý một dòng text từ ESP32."""
        if line.startswith("FB,"):
            parts = line[3:].split(",")
            if len(parts) >= 3:
                try:
                    raw0 = float(parts[0])
                    raw1 = float(parts[1])
                    raw2 = float(parts[2])
                    if not self.esp32_ready:
                        self.esp32_ready = True
                        self.status_message = "ESP32 READY qua feedback"
                        if self.pending_closed_loop and not self.closed_loop_control:
                            print("[TWAI] Feedback READY, tự vào Closed Loop theo yêu cầu trước đó.")
                            self.enter_closed_loop()
                    with self.data_lock:
                        self._raw_pos[0] = raw0
                        self._raw_pos[1] = raw1
                        self._raw_pos[2] = raw2
                        new_pos0 = self._rev_to_deg(raw0, 0)
                        new_pos1 = self._rev_to_deg(raw1, 1)
                        new_pos2 = self._rev_to_deg(raw2, 2)
                        now = time.time()
                        now_pc = time.perf_counter()
                        if self._fb_have_prev and self._fb_prev_pc is not None:
                            dt_fb = now_pc - self._fb_prev_pc
                            dt_fb = max(min(dt_fb, 0.25), 1e-4)
                            v0_raw = (new_pos0 - self._fb_prev_p[0]) / dt_fb
                            v1_raw = (new_pos1 - self._fb_prev_p[1]) / dt_fb
                            v2_raw = (new_pos2 - self._fb_prev_p[2]) / dt_fb
                            lim = self._vel_max_deg_s
                            v0_raw = max(min(v0_raw, lim), -lim)
                            v1_raw = max(min(v1_raw, lim), -lim)
                            v2_raw = max(min(v2_raw, lim), -lim)
                            self._update_vel_estimates_locked(now, v0_raw, v1_raw, v2_raw)
                        self._fb_prev_pc = now_pc
                        self._fb_prev_p[0] = new_pos0
                        self._fb_prev_p[1] = new_pos1
                        self._fb_prev_p[2] = new_pos2
                        self._fb_have_prev = True
                        self.pos[0] = new_pos0
                        self.pos[1] = new_pos1
                        self.pos[2] = new_pos2
                        self.data.append((
                            now,
                            self.pos[0], self.pos[1], self.pos[2],
                            self.pos_set[0], self.pos_set[1], self.pos_set[2]
                        ))
                except ValueError:
                    pass

        elif line.startswith("READY"):
            self.esp32_ready = True
            self.status_message = "ESP32 READY — có thể vào Closed Loop"
            print(f"[TWAI] {line}")
            if self.pending_closed_loop and not self.closed_loop_control:
                print("[TWAI] READY nhận được, tự vào Closed Loop theo yêu cầu trước đó.")
                self.enter_closed_loop()

        elif line.startswith("STATUS"):
            self.status_message = line
            print(f"[TWAI] {line}")

        elif line.startswith("WARN"):
            self.status_message = line
            print(f"[TWAI] {line}")

        elif line.startswith("INFO"):
            self.status_message = line
            print(f"[TWAI] {line}")

        elif line.startswith("ERROR"):
            self.error = True
            self.status_message = line
            print(f"[TWAI] {line}")

    # ════════════════════════════════════════════════════════════════════════
    # GUI-facing control methods
    # ════════════════════════════════════════════════════════════════════════

    def set_target(self, motor_id: int, pos_deg: float):
        """
        Đặt setpoint cho một motor (đơn vị: degrees).
        Không tự bật motion ngay; chỉ arm cho lần Move tiếp theo.
        """
        with self.data_lock:
            now = time.perf_counter()
            dt = max(now - self._last_set_ts, 1e-3)
            self.vel_set[motor_id] = (pos_deg - self._last_pos_set[motor_id]) / dt
            self.pos_set[motor_id] = pos_deg
            self._last_pos_set[motor_id] = pos_deg
            self._last_set_ts = now
            self._setpoint_dirty = True

    def set_both_targets(self, pos0_deg: float, pos1_deg: float, pos2_deg: float):
        """Đặt setpoint cho cả 3 motor cùng lúc, nhưng không tự chạy."""
        with self.data_lock:
            self.pos_set[0] = pos0_deg
            self.pos_set[1] = pos1_deg
            self.pos_set[2] = pos2_deg
            self._setpoint_dirty = True
            self.motion_armed = False
            self._motion_hold = False
            self._home_pending = False

    def get_data(self):
        """Trả snapshot của data buffer cho GUI plot."""
        with self.data_lock:
            return list(self.data)

    def get_pos(self):
        """Trả vị trí hiện tại (degrees) của cả 3 motor."""
        with self.data_lock:
            return self.pos[0], self.pos[1], self.pos[2]

    def get_setpoints(self):
        """Mục tiêu điều khiển hiện tại (deg): theo spline khi đang RUNNING."""
        with self.data_lock:
            return self.pos_set[0], self.pos_set[1], self.pos_set[2]

    # ── Compatibility shims cho guicontroller.py ────────────────────────────
    def update_ctrlElms(self, *ctrlElms):
        """
        Cập nhật control elements từ GUI.
        ctrlElms = (target_deg_m0, target_deg_m1, target_deg_m2, Kp, Kd, ctrl_bw, enc_bw)
        Chỉ lưu target; chỉ arm motion khi Move được nhấn từ GUI.
        """
        with self.data_lock:
            if len(ctrlElms) >= 3:
                new0 = float(ctrlElms[0])
                new1 = float(ctrlElms[1])
                new2 = float(ctrlElms[2])
                if getattr(self, "debug_sign_trace", False):
                    print(f"[TWAI][TRACE] update_ctrlElms input target=({new0:.6f}, {new1:.6f}, {new2:.6f})")
                for axis, new_target in enumerate((new0, new1, new2)):
                    if not self.locked_axes[axis]:
                        self._set_target_state(axis, new_target)
                    else:
                        self.pos_set[axis] = self.pos[axis]
                        self.vel_set[axis] = 0.0
                        self.acc_set[axis] = 0.0
                self._refresh_traj_refs_locked(time.time())
                self._last_pos_set[0] = new0
                self._last_pos_set[1] = new1
                self._last_pos_set[2] = new2
                self._last_set_ts = time.perf_counter()
                self._setpoint_dirty = True
                self.motion_armed = False
                self._motion_hold = False
                self._home_pending = False
                self._last_motion_targets[0] = new0
                self._last_motion_targets[1] = new1
                self._last_motion_targets[2] = new2
                if getattr(self, "debug_sign_trace", False):
                    print(f"[TWAI][TRACE] update_ctrlElms pos_set=({self.pos_set[0]:.6f}, {self.pos_set[1]:.6f}, {self.pos_set[2]:.6f})")
            if len(ctrlElms) >= 9:
                self.Kp_axes = [float(ctrlElms[3]), float(ctrlElms[4]), float(ctrlElms[5])]
                self.Kd_axes = [float(ctrlElms[6]), float(ctrlElms[7]), float(ctrlElms[8])]
                self.Kp3 = tuple(self.Kp_axes)
                self.Kd3 = tuple(self.Kd_axes)
            elif len(ctrlElms) >= 5:
                kp = float(ctrlElms[3])
                kd = float(ctrlElms[4])
                self.Kp_axes = [kp, kp, kp]
                self.Kd_axes = [kd, kd, kd]
                self.Kp3 = tuple(self.Kp_axes)
                self.Kd3 = tuple(self.Kd_axes)
            if len(ctrlElms) >= 7:
                self.ctrl_bandwidth = float(ctrlElms[5])
                self.enc_bandwidth  = float(ctrlElms[6])

    def apply_gui_targets_deg(self, p0_deg: float, p1_deg: float, p2_deg: float):
        
        with self.data_lock:
            if getattr(self, "debug_sign_trace", False):
                print(f"[TWAI][TRACE] apply_gui_targets_deg input=({p0_deg:.6f}, {p1_deg:.6f}, {p2_deg:.6f})")
            if getattr(self, "direct_setpoint_mode", False):
                self.pos_set[0] = float(p0_deg)
                self.pos_set[1] = float(p1_deg)
                self.pos_set[2] = float(p2_deg)
                self.vel_set[0] = 0.0
                self.vel_set[1] = 0.0
                self.vel_set[2] = 0.0
                self.acc_set[0] = 0.0
                self.acc_set[1] = 0.0
                self.acc_set[2] = 0.0
                self._last_pos_set[0] = float(p0_deg)
                self._last_pos_set[1] = float(p1_deg)
                self._last_pos_set[2] = float(p2_deg)
                self._last_set_ts = time.perf_counter()
                self._last_motion_targets[0] = float(p0_deg)
                self._last_motion_targets[1] = float(p1_deg)
                self._last_motion_targets[2] = float(p2_deg)
                if getattr(self, "debug_sign_trace", False):
                    print(f"[TWAI][TRACE] apply_gui_targets_deg DIRECT pos_set=({self.pos_set[0]:.6f}, {self.pos_set[1]:.6f}, {self.pos_set[2]:.6f})")
            else:
                self._set_target_state(0, float(p0_deg))
                self._set_target_state(1, float(p1_deg))
                self._set_target_state(2, float(p2_deg))
                self._motion_clock_start()
                self._refresh_traj_refs_locked(time.time())
                self._last_pos_set[0] = float(p0_deg)
                self._last_pos_set[1] = float(p1_deg)
                self._last_pos_set[2] = float(p2_deg)
                self._last_set_ts = time.perf_counter()
                self._last_motion_targets[0] = float(p0_deg)
                self._last_motion_targets[1] = float(p1_deg)
                self._last_motion_targets[2] = float(p2_deg)
                if getattr(self, "debug_sign_trace", False):
                    print(f"[TWAI][TRACE] apply_gui_targets_deg pos_set=({self.pos_set[0]:.6f}, {self.pos_set[1]:.6f}, {self.pos_set[2]:.6f})")
    def update_loadParms(self, *loadParms):
        """Cập nhật load parameters từ GUI (giữ tương thích)."""
        with self.data_lock:
            if len(loadParms) >= 1: self.ext_load        = float(loadParms[0])
            if len(loadParms) >= 2: self.hanger_distance  = float(loadParms[1])
            if len(loadParms) >= 3: self.coul_friction    = float(loadParms[2])
            if len(loadParms) >= 4: self.visc_friction     = float(loadParms[3])
            if len(loadParms) >= 5: self.max_torque        = float(loadParms[4])
            self._recalc_plant_mass_inertia()

    def clear_error(self):
        self.error = False

    def set_torque_mode(self, enabled: bool = True):
        self.use_torque_mode = enabled
        if self.esp32_ready:
            self._send_simple_cmd("TORQUE" if enabled else "POSITION")

    def go_home(self):
        """Đưa cả 3 motor về gốc index bằng lệnh HOME của ESP32."""
        with self.data_lock:
            self.motion_armed = False
            self._motion_hold = False
            self._home_pending = True
            self._setpoint_dirty = False
            self._was_motion_armed = False
            self._motion_time_active = False
            self._tau_out[0] = self._tau_out[1] = self._tau_out[2] = 0.0
        self._send_simple_cmd("HOME")
        self.status_message = "Đang home về index"

    # ════════════════════════════════════════════════════════════════════════
    # Main Thread Loop
    # ════════════════════════════════════════════════════════════════════════

    def run(self):
        """Vòng lặp chính: ~100Hz — đọc Serial và gửi lệnh position."""
        print("[TWAI] Thread khởi động.")

        while not self._stop_event.is_set():
            t_start = time.perf_counter()

            # ── Kết nối nếu chưa kết nối ─────────────────────────────────
            if not self.connected:
                self.connect()
                if not self.connected:
                    self._stop_event.wait(0.5)
                    continue

            # ── ESTOP: không làm gì, chỉ chờ ─────────────────────────────
            if self._estop_event.is_set():
                self._stop_event.wait(0.05)
                continue

            try:
                # 1. Đọc và parse phản hồi từ ESP32
                self._process_serial()

                # Poll trạng thái bridge định kỳ
                now = time.time()
                if now - self._last_status_poll_ts >= 2.0:
                    self._send_simple_cmd("STATUS")
                    self._last_status_poll_ts = now

                # 2. Gửi lệnh tới ESP32 chỉ khi đã arm motion
                if self.is_controlable() and self.motion_armed:
                    now_wall = time.time()
                    if self.friction_only_mode:
                        axis = self.friction_only_axis if self.friction_only_axis is not None else 0
                        tau = self.friction_only_torque
                        self._send_torque(0, 0.0)
                        self._send_torque(1, 0.0)
                        self._send_torque(2, 0.0)
                        tau_motor = self._joint_to_motor_torque(axis, tau)
                        self._send_torque(axis, tau_motor)
                        self.torque_set[0] = 0.0
                        self.torque_set[1] = 0.0
                        self.torque_set[2] = 0.0
                        self.torque_set[axis] = tau_motor
                        if now_wall - self._last_torque_log_ts >= 1.0:
                            print(f"[TWAI] Friction-only mode axis={axis} tau={tau:.6f}")
                            self._last_torque_log_ts = now_wall
                        continue
                    arm_edge = self.motion_armed and not self._was_motion_armed
                    if arm_edge:
                        self._motion_clock_start()

                    tau0 = tau1 = 0.0
                    ok_done = False
                    with self.data_lock:
                        self._refresh_traj_refs_locked(now_wall)
                        p0_set = self.pos_set[0]
                        p1_set = self.pos_set[1]
                        p2_set = self.pos_set[2]
                        p0 = self.pos[0]
                        p1 = self.pos[1]
                        p2 = self.pos[2]
                        v0 = self.vel[0]
                        v1 = self.vel[1]
                        v2 = self.vel[2]
                        err0 = p0_set - p0
                        err1 = p1_set - p1
                        err2 = p2_set - p2
                        t_prog = max(now_wall - self._motion_t0, 0.0)
                        done_trap0 = t_prog >= self.traj[0].total_time
                        done_trap1 = t_prog >= self.traj[1].total_time
                        done_trap2 = t_prog >= self.traj[2].total_time
                        pos_ok0 = abs(err0) <= self.target_tolerance_deg
                        pos_ok1 = abs(err1) <= self.target_tolerance_deg
                        pos_ok2 = abs(err2) <= self.target_tolerance_deg
                        vel_ok0 = abs(v0) <= self.target_tolerance_vel
                        vel_ok1 = abs(v1) <= self.target_tolerance_vel
                        vel_ok2 = abs(v2) <= self.target_tolerance_vel

                        ok_done = (
                            done_trap0
                            and done_trap1
                            and done_trap2
                            and pos_ok0
                            and pos_ok1
                            and pos_ok2
                            and vel_ok0
                            and vel_ok1
                            and vel_ok2
                        )

                        if ok_done:
                            self.torque_set[0] = 0.0
                            self.torque_set[1] = 0.0
                            self.torque_set[2] = 0.0
                        elif self.use_torque_mode:
                            active_axis = getattr(self, "single_axis_test", None)
                            self._dynamic_calculation_locked(active_axis=active_axis)
                            for axis, locked in enumerate(self.locked_axes):
                                if locked:
                                    self.torque_set[axis] = 0.0
                            tau0 = self.torque_set[0]
                            tau1 = self.torque_set[1]
                            tau2 = self.torque_set[2]
                        else:
                            pass

                    if ok_done:
                        self._motion_time_active = False
                        self._reset_torque_slew()
                        self._send_torque(0, 0.0)
                        self._send_torque(1, 0.0)
                        self._send_torque(2, 0.0)
                        self.motion_armed = False
                        self._was_motion_armed = False
                        self._motion_hold = True
                        self._setpoint_dirty = False
                        self.status_message = "Motion completed"
                    elif self.use_torque_mode:
                        tau0_joint = tau0
                        tau1_joint = tau1
                        tau2_joint = tau2
                        tau0_sent = self._slew_limited_torque(0, tau0_joint)
                        tau1_sent = self._slew_limited_torque(1, tau1_joint)
                        tau2_sent = self._slew_limited_torque(2, tau2_joint)
                        tau0_motor = self._joint_to_motor_torque(0, tau0_sent)
                        tau1_motor = self._joint_to_motor_torque(1, tau1_sent)
                        tau2_motor = self._joint_to_motor_torque(2, tau2_sent)
                        self._tau_sent = (tau0_sent, tau1_sent, tau2_sent)
                        self._send_torque(0, tau0_motor)
                        self._send_torque(1, tau1_motor)
                        self._send_torque(2, tau2_motor)
                        if now_wall - self._last_torque_log_ts >= 1.0:
                            print(
                                "[TWAI] CTC torque cmd "
                                f"tau_raw=({self._last_tau_raw[0]:.4f}, {self._last_tau_raw[1]:.4f}, {self._last_tau_raw[2]:.4f}) "
                                f"tau_joint_cmd=({tau0_sent:.4f}, {tau1_sent:.4f}, {tau2_sent:.4f}) "
                                f"tau_motor_cmd=({tau0_motor:.4f}, {tau1_motor:.4f}, {tau2_motor:.4f}) "
                                f"gear=({self.gear_ratios[0]:.1f}, {self.gear_ratios[1]:.1f}, {self.gear_ratios[2]:.1f}) "
                                f"eff=({self.motor_efficiency[0]:.2f}, {self.motor_efficiency[1]:.2f}, {self.motor_efficiency[2]:.2f}) "
                                f"p_ref0={p0_set:.2f}, p_ref1={p1_set:.2f}, p_ref2={p2_set:.2f}, "
                                f"err_trk0={err0:.2f}deg, err_trk1={err1:.2f}deg, err_trk2={err2:.2f}deg"
                            )
                            self._last_torque_log_ts = now_wall
                    else:
                        with self.data_lock:
                            pv0_set = self.vel_set[0]
                            pv1_set = self.vel_set[1]
                            pv2_set = self.vel_set[2]
                        rev0 = self._deg_to_rev(p0_set, 0)
                        rev1 = self._deg_to_rev(p1_set, 1)
                        rev2 = self._deg_to_rev(p2_set, 2)
                        vel0 = pv0_set * self.gear_ratios[0] / 360.0 if hasattr(self, "gear_ratios") else pv0_set * self.gear_ratio / 360.0
                        vel1 = pv1_set * self.gear_ratios[1] / 360.0 if hasattr(self, "gear_ratios") else pv1_set * self.gear_ratio / 360.0
                        vel2 = pv2_set * self.gear_ratios[2] / 360.0 if hasattr(self, "gear_ratios") else pv2_set * self.gear_ratio / 360.0
                        self._send_position(0, rev0, vel0)
                        self._send_position(1, rev1, vel1)
                        self._send_position(2, rev2, vel2)
                elif self.is_controlable() and not self.motion_armed:
                    self._motion_time_active = False
                    self._reset_torque_slew()
                    self._send_torque(0, 0.0)
                    self._send_torque(1, 0.0)
                    self._send_torque(2, 0.0)
                self._was_motion_armed = self.motion_armed

            except Exception as e:
                print(f"[TWAI] Lỗi vòng lặp: {e}")
                self.connected = False
                self.error     = True
                self._stop_event.wait(1.0)

            # ── Giữ ~100Hz ────────────────────────────────────────────────
            elapsed = time.perf_counter() - t_start
            sleep_t = 0.01 - elapsed
            if sleep_t > 0:
                self._stop_event.wait(sleep_t)

        print("[TWAI] Thread dừng.")
