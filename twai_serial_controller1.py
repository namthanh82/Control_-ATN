import threading
import time
import math
import serial
import serial.tools.list_ports
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kinematic import get_acc_jerk
from trajectory import SplineTrajectory, QuinticTrajectory
from ctc_3dof import CTC3Gains, CTC3Model, JointParams, ctc_3dof, ctc_3dof_components

# ── Helpers ────────────────────────────────────────────────────────────────────
def _lp_alpha_from_fc(fc_hz: float, fs_hz: float) -> float:
    """Compute IIR 1-pole LP alpha from corner frequency.
    y[n] = alpha * x[n] + (1-alpha) * y[n-1]
    alpha = 2π·fc / (2π·fc + fs)
    fs = loop frequency (Hz), fc = desired -3dB corner (Hz).
    """
    return (2.0 * math.pi * fc_hz) / (2.0 * math.pi * fc_hz + fs_hz)

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
        baudrate: int = 230400,
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
        self._last_fb_time     = 0.0

        # ── Threading primitives ─────────────────────────────────────────
        self.data_lock   = threading.Lock()
        self._stop_event  = threading.Event()
        self._estop_event = threading.Event()
        # ── Offset / encoder state ───────────────────────────────────────
        self.offset_rev = [0.0, 0.0, 0.0]      
        self.motor_pos_rev = [0.0, 0.0, 0.0]   
        self.joint_pos_deg = [0.0, 0.0, 0.0]    
        self.joint_offset_deg = [0.0, 0.0, 0.0]  
        self.model_home_deg = [0.0, -90.0, 0.0]
        self.use_joint_feedback = True         # khi True, CTC dùng joint encoder làm feedback chính
        self.debug_sign_trace = True         # bật in [TWAI][TRACE] CTC/trajectory
        self.debug_serial_verbose = False    # True = in mọi dòng nhận từ ESP32 (tắt khi chạy ổn định)
        self.debug_timing = True            # True = in thống kê timing vòng feedback mỗi 1s
        self._trace_log_interval_s = 0.25      # giới hạn trace vòng lặp ~100Hz → tối đa ~4 dòng/giây
        self._last_trace_loop_pc = 0.0
        # Timing stats (reset mỗi lần bắt đầu run)
        self._timing_last_esp_ms = 0
        self._timing_last_pc_s = 0.0
        self._timing_dts = []              # list[float] loop_dt (ms)
        self._timing_latency = []          # list[float] esp→PC latency (ms)
        self._timing_last_report = 0.0

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
        self.max_vel   = 25 # °/s — giới hạn cho param_calc của spline (GUI có thể mở rộng sau)
        self._motion_t0           = -math.inf
        self._motion_time_active  = False
        self._was_motion_armed    = False
        self.tor_coef  = 1.0
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
        self._vel_max_deg_s = 800.0
        # ── LP filter cho qdot (tốc độ phản hồi thực) ───────────────────────
        # fc: corner frequency (Hz). Lọc nhiễu encoder trước khi tính Kd·de.
        #   50-70 Hz: lọc mạnh, mượt, dùng với Kd cao
        #   80-120 Hz: lọc nhẹ, phản ứng nhanh, dùng với Kd thấp
        #   công thức: alpha = 2π·fc / (2π·fc + fs), fs = 100 Hz (loop chính)
        self._vel_lp_hz = 80.0
        self._vel_lp_alpha = _lp_alpha_from_fc(self._vel_lp_hz, 100.0)
        self._vel_lp_prev = [0.0, 0.0, 0.0]   # prev LP state
        # ── LP filter cho qdot_d (feedforward velocity từ trajectory) ─────────
        # Giữ riêng để có thể lọc trajectory mà không ảnh hưởng vel phản hồi.
        self._vel_set_lp_hz = 80.0
        self._vel_set_lp_alpha = _lp_alpha_from_fc(self._vel_set_lp_hz, 100.0)
        self._vel_set_lp_prev = [0.0, 0.0, 0.0]

        self.torque_set = [0.0, 0.0, 0.0]
        self._last_tau_raw = (0.0, 0.0, 0.0)
        self.use_torque_commands = True

        # ── Control params (dùng cho GUI control panel) ──────────────────
        self.Kp_axes = [3.0, 6.0, 3.0]
        self.Kd_axes = [1.0, 0.5, 1.0]
        self.Kp3 = tuple(self.Kp_axes)
        self.Kd3 = tuple(self.Kd_axes)
        self.ctrl_bandwidth = 1200
        self.enc_bandwidth  = 100
        self.max_torque     = 1.5 # torque limit used for saturation in bridge mode
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
        self.data: deque = deque(maxlen=800)

        # ── Serial receive line buffer ───────────────────────────────────
        self._line_buf = ""
        self._last_status_poll_ts = 0.0
        self._last_torque_log_ts = 0.0

        # Giới hạn tốc độ đổi mô-men (Nm mỗi bước ~10ms) để tránh flip ±max_torque quá nhanh → lắc
        self._tau_out = [0.0, 0.0, 0.0]
        self.motor_efficiency = (1, 1, 1)
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
        self._vel_lp_prev = [0.0, 0.0, 0.0]
        self._vel_set_lp_prev = [0.0, 0.0, 0.0]
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
            self.vel[2] = 0.0
            self.acc_set[0] = 0.0
            self.acc_set[1] = 0.0
            self.acc_set[2] = 0.0
            self._last_motion_targets[0] = 0.0
            self._last_motion_targets[1] = 0.0
            self._last_motion_targets[2] = 0.0
            self.joint_offset_deg = [0.0, 0.0, 0.0]
            self._clear_vel_fit_buffers_locked()
            self._reset_torque_slew()
            self._timing_dts = []
            self._timing_latency = []
            self._timing_last_report = 0.0
            self._timing_last_pc_s = 0.0
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
        Lưu vị trí hiện tại làm home (0°):
        - motor encoder: offset_rev từ FB
        - joint encoder: joint_offset_deg từ FBJ (khi use_joint_feedback)
        """
        with self.data_lock:
            self.offset_rev[0] = self.motor_pos_rev[0]
            self.offset_rev[1] = self.motor_pos_rev[1]
            self.offset_rev[2] = self.motor_pos_rev[2]
            for i in range(3):
                # joint_pos_deg đã là góc tương đối home; cộng offset cũ = góc thô từ firmware
                self.joint_offset_deg[i] = self.joint_pos_deg[i] + self.joint_offset_deg[i]
                self.joint_pos_deg[i] = 0.0
                self.pos[i] = 0.0
                self.pos_set[i] = 0.0
                self.vel_set[i] = 0.0
                self.acc_set[i] = 0.0
                self._last_pos_set[i] = 0.0
                self._last_motion_targets[i] = 0.0
            self.isOffset = True
            self.motion_armed = False
            self._motion_hold = False
            self._motion_time_active = False
            self._motion_t0 = -math.inf
            self._was_motion_armed = False
            self._fb_have_prev = False
            self._fb_prev_pc = None
            self._fb_prev_p = [0.0, 0.0, 0.0]
            self.vel[0] = self.vel[1] = self.vel[2] = 0.0
            self._tau_out[0] = self._tau_out[1] = self._tau_out[2] = 0.0
            self._clear_vel_fit_buffers_locked()
            self._timing_dts = []
            self._timing_latency = []
            self._timing_last_report = 0.0
            self._timing_last_pc_s = 0.0
            for traj in self.traj:
                traj.reset()
        self.status_message = "Offset set — vị trí hiện tại = home (0°)"
        print(f"[TWAI] Offset/home set: motor_rev=({self.offset_rev[0]:.4f}, {self.offset_rev[1]:.4f}, {self.offset_rev[2]:.4f}), "
              f"joint_offset_deg=({self.joint_offset_deg[0]:.3f}, {self.joint_offset_deg[1]:.3f}, {self.joint_offset_deg[2]:.3f})")

    # ════════════════════════════════════════════════════════════════════════
    # Unit conversion helpers
    # ════════════════════════════════════════════════════════════════════════

    def _rev_to_deg(self, rev: float, motor_id: int) -> float:
        gear = self.gear_ratios[motor_id] if hasattr(self, "gear_ratios") else self.gear_ratio
        return (rev - self.offset_rev[motor_id]) * 360.0 * gear + self.start_pos

    def _deg_to_rev(self, deg: float, motor_id: int) -> float:
        """Degrees → ODrive raw rev (motor-side only)."""
        gear = self.gear_ratios[motor_id] if hasattr(self, "gear_ratios") else self.gear_ratio
        return (deg - self.start_pos) / 360.0 / gear + self.offset_rev[motor_id]

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

        Sau poly-fit (smooth), đầu ra được LP-filter thêm một lớp nữa qua IIR 1-pole
        với corner frequency `_vel_lp_hz` Hz (mặc định 80 Hz) để triệt nhiễu encoder
        high-frequency trước khi tính Kd·de.
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
                    vf_clipped = max(min(float(vf), lim), -lim)
                except ValueError:
                    vf_clipped = v_inst
                a = self._vel_lp_alpha
                self.vel[i] = a * vf_clipped + (1.0 - a) * self._vel_lp_prev[i]
                self._vel_lp_prev[i] = self.vel[i]
            else:
                a = self._vel_lp_alpha
                self.vel[i] = a * v_inst + (1.0 - a) * self._vel_lp_prev[i]
                self._vel_lp_prev[i] = self.vel[i]

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
        start_p = self.pos[motor_id]
        self.traj[motor_id].param_calc(start_p, target_deg, self.max_vel)
        if self.debug_sign_trace:
            print(
                f"[TWAI][TRACE] _set_target_state joint={motor_id} traj={type(self.traj[motor_id]).__name__} "
                f"start_p={start_p:.6f} target_deg={target_deg:.6f} max_vel={self.max_vel:.6f}"
            )

    def _trace_loop_due(self) -> bool:
        """Chỉ cho phép in trace vòng điều khiển vài lần/giây (tránh spam ~100Hz)."""
        if not self.debug_sign_trace:
            return False
        now = time.perf_counter()
        if now - self._last_trace_loop_pc < self._trace_log_interval_s:
            return False
        self._last_trace_loop_pc = now
        return True

    def _motion_clock_start(self):
        self._motion_t0 = time.time()
        self._motion_time_active = True
        # Reset LP filter state để tránh spike từ giá trị cũ khi bắt đầu motion mới.
        self._vel_lp_prev = [0.0, 0.0, 0.0]
        self._vel_set_lp_prev = [0.0, 0.0, 0.0]

    def _refresh_traj_refs_locked(self, now: float, trace: bool = False):
        """Cập nhật pos_set / vel_set / acc_set theo thời gian đã trôi của spline."""
        if self._motion_time_active:
            t_prog = max(now - self._motion_t0, 0.0)
        else:
            t_prog = 0.0
        for i in (0, 1, 2):
            p_des, v_des, a_des = self.traj[i].desired_state(t_prog)
            self.pos_set[i], self.vel_set[i], self.acc_set[i] = p_des, v_des, a_des
        if trace:
            print(
                f"[TWAI][TRACE] traj t={t_prog:.3f}s "
                f"pos_set=({self.pos_set[0]:.3f}, {self.pos_set[1]:.3f}, {self.pos_set[2]:.3f}) "
                f"vel_set=({self.vel_set[0]:.3f}, {self.vel_set[1]:.3f}, {self.vel_set[2]:.3f})"
            )

    def _dynamic_calculation_locked(self, active_axis: int | None = None, trace: bool = False):
        """Compute 3-DOF torque commands using the M+C+G CTC model."""
        gains = CTC3Gains(kp=self.Kp3, kd=self.Kd3)
        self.model3.torque_scale = self.tor_coef
        # Đồng bộ hệ số bù mô hình mỗi tick (cho phép tune runtime).
        q_source = self.joint_pos_deg if self.use_joint_feedback else self.pos
        # Cộng offset home vật lý để CTC thấy góc thật (bù trọng lực đúng pha).
        # Sai số e=qd-q không đổi vì cùng cộng offset cho cả q và qd.
        q = [(q_source[i] + self.model_home_deg[i]) * DEG2RAD for i in range(3)]
        qd = [(self.pos_set[i] + self.model_home_deg[i]) * DEG2RAD for i in range(3)]
        qdot = [self.vel[i] * DEG2RAD for i in range(3)]
        # LP filter cho qdot_d (feedforward vel từ trajectory).
        # vel_set có thể có chatter từ spline derivative — lọc để Kd term không bị nhiễu.
        qdot_d = []
        for i in range(3):
            vs_raw = self.vel_set[i] * DEG2RAD
            a = self._vel_set_lp_alpha
            vs_filt = a * vs_raw + (1.0 - a) * self._vel_set_lp_prev[i]
            self._vel_set_lp_prev[i] = vs_filt
            qdot_d.append(vs_filt)
        qddot_d = [self.acc_set[i] * DEG2RAD for i in range(3)]

        # Smooth-startup: t_prog tính từ lúc bắt đầu motion để blend G_hold→G_ff.
        if self._motion_time_active:
            startup_t = max(time.time() - self._motion_t0, 0.0)
        else:
            startup_t = 0.0

        comp = ctc_3dof_components(qd, q, qdot_d, qdot, qddot_d, gains, self.model3, startup_t=startup_t)
        tau = comp["tau"]
        self._last_tau_raw = tuple(tau)
        for i in range(3):
            if active_axis is None or i == active_axis:
                self.torque_set[i] = float(tau[i])
            else:
                self.torque_set[i] = 0.0
        if trace:
            G = comp["g"]
            w = comp["startup_w"]
            print(
                f"[TWAI][TRACE] ctc t_prog={startup_t:.3f}s w={w:.2f} "
                f"q_deg=({q_source[0]:.3f},{q_source[1]:.3f},{q_source[2]:.3f}) "
                f"qd_deg=({self.pos_set[0]:.3f},{self.pos_set[1]:.3f},{self.pos_set[2]:.3f}) "
                f"G=({G[0]:.3f},{G[1]:.3f},{G[2]:.3f}) "
                f"tau=({self.torque_set[0]:.3f},{self.torque_set[1]:.3f},{self.torque_set[2]:.3f})"
            )

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
        # Option B: ramp ở ODrive (INPUT_MODE_TORQUE_RAMP), host chỉ clip theo max_torque.
        out = max(min(float(tau_des), self.max_torque), -self.max_torque)


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
                    if self.debug_serial_verbose:
                        print(f"[TWAI] RX raw: '{line}'")
                    self._parse_line(line)
        except Exception as e:
            print(f"[TWAI] Lỗi đọc Serial: {e}")
            self.connected = False

    def _parse_line(self, line: str):
        """Xử lý một dòng text từ ESP32.

        Firmware gửi:
          FB,ts_ms,mot0_rev,mot1_rev,mot2_rev,joint0_deg,joint1_deg,joint2_deg
        """
        if self.debug_serial_verbose:
            print(f"[TWAI] <<< {line}")
        if line.startswith("FB,"):
            parts = line[3:].split(",")
            if len(parts) < 7:
                print(f"[TWAI] FB format error: {len(parts)} values — '{line}'")
                return
            try:
                ts_ms = int(parts[0])       # ESP32 millis() timestamp
                raw0  = float(parts[1])
                raw1  = float(parts[2])
                raw2  = float(parts[3])
                j0    = float(parts[4])
                j1    = float(parts[5])
                j2    = float(parts[6])
                if self.debug_serial_verbose:
                    print(f"[TWAI] RAW motor1={raw1:.6f}  joint1={j1:.4f}")
            except ValueError:
                if self.debug_serial_verbose:
                    print(f"[TWAI] FB ValueError: parts={parts!r}")
                return

            self._last_fb_time = time.time()
            if not self.esp32_ready:
                self.esp32_ready = True
                self.status_message = "ESP32 READY"
                if self.pending_closed_loop and not self.closed_loop_control:
                    print("[TWAI] Feedback READY, tự vào Closed Loop.")
                    self.enter_closed_loop()
            self.error = False

            with self.data_lock:
                now = time.time()
                now_pc = time.perf_counter()

                # Motor encoder positions (for reference / calibration)
                self.motor_pos_rev[0] = raw0
                self.motor_pos_rev[1] = raw1
                self.motor_pos_rev[2] = raw2

                # Joint angles — subtract offset so home = 0°
                j0_off = j0 - self.joint_offset_deg[0]
                j1_off = j1 - self.joint_offset_deg[1]
                j2_off = j2 - self.joint_offset_deg[2]

                # Motor encoder in degrees (always calculated from raw rev)
                mot0_deg = self._rev_to_deg(raw0, 0)
                mot1_deg = self._rev_to_deg(raw1, 1)
                mot2_deg = self._rev_to_deg(raw2, 2)

                # Decide which source feeds pos and vel
                if self.use_joint_feedback:
                    self.pos[0] = j0_off
                    self.pos[1] = j1_off
                    self.pos[2] = j2_off
                    self.joint_pos_deg[0] = j0_off
                    self.joint_pos_deg[1] = j1_off
                    self.joint_pos_deg[2] = j2_off
                else:
                    new_p0 = self._rev_to_deg(raw0, 0)
                    new_p1 = self._rev_to_deg(raw1, 1)
                    new_p2 = self._rev_to_deg(raw2, 2)
                    self.pos[0] = new_p0
                    self.pos[1] = new_p1
                    self.pos[2] = new_p2

                # Velocity from whichever source is active
                p0, p1, p2 = self.pos[0], self.pos[1], self.pos[2]
                if self._fb_have_prev and self._fb_prev_pc is not None:
                    dt = now_pc - self._fb_prev_pc
                    dt = max(min(dt, 0.25), 1e-4)
                    v0 = (p0 - self._fb_prev_p[0]) / dt
                    v1 = (p1 - self._fb_prev_p[1]) / dt
                    v2 = (p2 - self._fb_prev_p[2]) / dt
                    lim = self._vel_max_deg_s
                    self._update_vel_estimates_locked(
                        now,
                        max(min(v0, lim), -lim),
                        max(min(v1, lim), -lim),
                        max(min(v2, lim), -lim),
                    )
                self._fb_prev_pc = now_pc
                self._fb_prev_p[0] = p0
                self._fb_prev_p[1] = p1
                self._fb_prev_p[2] = p2
                self._fb_have_prev = True

                self.data.append((
                    now,
                    self.pos[0], self.pos[1], self.pos[2],      # 1-3: active feedback (joint or motor deg)
                    self.pos_set[0], self.pos_set[1], self.pos_set[2],  # 4-6: setpoint
                    self.acc_set[0], self.acc_set[1], self.acc_set[2],  # 7-9: acc set
                    self._tau_out[0], self._tau_out[1], self._tau_out[2],  # 10-12: torque out
                    self.torque_set[0], self.torque_set[1], self.torque_set[2],  # 13-15: torque set
                    mot0_deg, mot1_deg, mot2_deg,                # 16-18: motor encoder deg
                    j0 - self.joint_offset_deg[0],              # 19: raw joint0 - offset
                    j1 - self.joint_offset_deg[1],              # 20: raw joint1 - offset
                    j2 - self.joint_offset_deg[2],              # 21: raw joint2 - offset
                ))
                if self.debug_serial_verbose:
                    print(f"[TWAI] APPEND pos=({self.pos[0]:.3f},{self.pos[1]:.3f},{self.pos[2]:.3f}) "
                          f"offset=({self.joint_offset_deg[0]:.3f},{self.joint_offset_deg[1]:.3f},{self.joint_offset_deg[2]:.3f}) "
                          f"buf_len={len(self.data)}")

                # ── Timing stats ──────────────────────────────────────────────────
                if self.debug_timing:
                    latency_ms = (now * 1000.0) - ts_ms   # PC_recv_ms − ESP_send_ms
                    loop_dt_ms = (now * 1000.0) - self._timing_last_pc_s * 1000.0 if self._timing_last_pc_s else 0.0
                    self._timing_latency.append(latency_ms)
                    if loop_dt_ms > 0:
                        self._timing_dts.append(loop_dt_ms)
                    self._timing_last_pc_s = now
                    self._timing_last_esp_ms = ts_ms
                    # Report every ~1 second
                    if now - self._timing_last_report >= 1.0:
                        import statistics
                        dts = self._timing_dts[-500:] if len(self._timing_dts) > 500 else self._timing_dts
                        lat = self._timing_latency[-500:] if len(self._timing_latency) > 500 else self._timing_latency
                        if dts and lat:
                            print(f"[TWAI] Timing — loop_dt: avg={statistics.mean(dts):.2f}ms "
                                  f"min={min(dts):.2f} max={max(dts):.2f} "
                                  f"std={statistics.stdev(dts) if len(dts) > 1 else 0:.2f} | "
                                  f"latency: avg={statistics.mean(lat):.2f}ms "
                                  f"min={min(lat):.2f} max={max(lat):.2f} | "
                                  f"n={len(dts)}")
                        self._timing_last_report = now

        elif line.startswith("READY"):
            self.esp32_ready = True
            self.status_message = "ESP32 READY — có thể vào Closed Loop"
            print(f"[TWAI] {line}")
            if self.pending_closed_loop and not self.closed_loop_control:
                print("[TWAI] READY nhận được, tự vào Closed Loop.")
                self.enter_closed_loop()

        elif line.startswith("STATUS"):
            self.status_message = line
            if self.debug_serial_verbose:
                print(f"[TWAI] {line}")

        elif line.startswith("WARN"):
            if self.debug_serial_verbose:
                print(f"[TWAI] {line}")
            elif "feedback timeout" not in line and "RX queue full" not in line:
                self.status_message = line

        elif line.startswith("ERROR"):
            self.status_message = line
            if self.debug_serial_verbose:
                print(f"[TWAI] {line}")
            if "CAN init failed" in line or "heartbeat" in line:
                self.error = True

        elif line.startswith("LP ") or line.startswith("VEL_LP "):
            try:
                parts = line.split()
                fc_vel = float(parts[1])
                fc_set = float(parts[2]) if len(parts) >= 3 else fc_vel
                self.set_vel_lp_hz(fc_vel, fc_set)
                print(f"[TWAI] LP: vel_hz={self._vel_lp_hz:.1f} set_hz={self._vel_set_lp_hz:.1f}")
            except (IndexError, ValueError):
                print("[TWAI] Usage: LP <vel_hz> [set_hz]")


    def set_vel_lp_hz(self, fc_vel_hz: float, fc_set_hz: float | None = None):
        """Thay đổi corner frequency của LP filter tại runtime.

        Args:
            fc_vel_hz:  corner frequency cho qdot (vận tốc phản hồi thực).
                        Giảm xuống → lọc mạnh hơn, ít rung, phản ứng chậm hơn.
                        Tăng lên → lọc nhẹ, phản ứng nhanh hơn, có thể rung nhiều hơn.
                        Khuyến nghị: 40-100 Hz.
            fc_set_hz:  corner frequency cho qdot_d (feedforward từ trajectory).
                        Mặc định = fc_vel_hz nếu không truyền.
        """
        self._vel_lp_hz = float(fc_vel_hz)
        self._vel_lp_alpha = _lp_alpha_from_fc(self._vel_lp_hz, 100.0)
        self._vel_set_lp_hz = float(fc_set_hz) if fc_set_hz is not None else self._vel_lp_hz
        self._vel_set_lp_alpha = _lp_alpha_from_fc(self._vel_set_lp_hz, 100.0)
        print(f"[TWAI] LP updated: vel_hz={self._vel_lp_hz:.1f} (alpha={self._vel_lp_alpha:.4f}), "
              f"set_hz={self._vel_set_lp_hz:.1f} (alpha={self._vel_set_lp_alpha:.4f})")

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
                if self.debug_sign_trace:
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
                if self.debug_sign_trace:
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
        """Cập nhật mục tiêu khi RUNNING; chỉ tính lại spline khi target thực sự đổi."""
        p0_deg = float(p0_deg)
        p1_deg = float(p1_deg)
        p2_deg = float(p2_deg)
        with self.data_lock:
            if self.debug_sign_trace:
                print(f"[TWAI][TRACE] apply_gui_targets_deg input=({p0_deg:.6f}, {p1_deg:.6f}, {p2_deg:.6f})")

            unchanged = (
                math.isclose(p0_deg, self._last_motion_targets[0], abs_tol=1e-6)
                and math.isclose(p1_deg, self._last_motion_targets[1], abs_tol=1e-6)
                and math.isclose(p2_deg, self._last_motion_targets[2], abs_tol=1e-6)
            )
            if unchanged and self._motion_time_active:
                # Target không đổi: giữ đồng hồ spline, để run-loop tiếp tục qd theo thời gian.
                return

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
                if self.debug_sign_trace:
                    print(f"[TWAI][TRACE] apply_gui_targets_deg DIRECT pos_set=({self.pos_set[0]:.6f}, {self.pos_set[1]:.6f}, {self.pos_set[2]:.6f})")
            else:
                self._set_target_state(0, float(p0_deg))
                self._set_target_state(1, float(p1_deg))
                self._set_target_state(2, float(p2_deg))
                self._motion_clock_start()
                self._refresh_traj_refs_locked(time.time(), trace=self.debug_sign_trace)
                self._last_pos_set[0] = float(p0_deg)
                self._last_pos_set[1] = float(p1_deg)
                self._last_pos_set[2] = float(p2_deg)
                self._last_set_ts = time.perf_counter()
                self._last_motion_targets[0] = float(p0_deg)
                self._last_motion_targets[1] = float(p1_deg)
                self._last_motion_targets[2] = float(p2_deg)
                if self.debug_sign_trace:
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

            # ── Heartbeat check: reconnect nếu không nhận feedback > 3s ───────
            if self.esp32_ready and (time.time() - self._last_fb_time) > 3.0:
                print("[TWAI] WARN: mất feedback >3s, reset input buffer và thử lại...")
                self.esp32_ready = False
                self._last_fb_time = time.time()
                if self.ser:
                    try:
                        self.ser.reset_input_buffer()
                    except Exception:
                        pass

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

                    tau0 = tau1 = tau2 = 0.0
                    ok_done = False
                    trace_now = self._trace_loop_due()
                    with self.data_lock:
                        self._refresh_traj_refs_locked(now_wall, trace=trace_now)
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
                            self._dynamic_calculation_locked(active_axis=active_axis, trace=trace_now)
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
                        tau0_motor = self._joint_to_motor_torque(0, tau0)
                        tau1_motor = self._joint_to_motor_torque(1, tau1)
                        tau2_motor = self._joint_to_motor_torque(2, tau2)
                        tau0_sent = self._slew_limited_torque(0, tau0_motor)
                        tau1_sent = self._slew_limited_torque(1, tau1_motor)
                        tau2_sent = self._slew_limited_torque(2, tau2_motor)
                        self._tau_sent = (tau0_sent, tau1_sent, tau2_sent)
                        self._send_torque(0, tau0_sent)
                        self._send_torque(1, tau1_sent)
                        self._send_torque(2, tau2_sent)
                        if trace_now:
                            print(
                                f"[TWAI][TRACE] tau_driver sent_Nm=({tau0_sent:.4f},{tau1_sent:.4f},{tau2_sent:.4f}) "
                                f"motor_pre_slew=({tau0_motor:.4f},{tau1_motor:.4f},{tau2_motor:.4f})"
                            )
                        if now_wall - self._last_torque_log_ts >= 1.0:
                            print(
                                "[TWAI] CTC torque cmd "
                                f"tau_raw=({self._last_tau_raw[0]:.4f}, {self._last_tau_raw[1]:.4f}, {self._last_tau_raw[2]:.4f}) "
                                f"tau_joint_cmd=({tau0:.4f}, {tau1:.4f}, {tau2:.4f}) "
                                f"tau_motor_cmd=({tau0_sent:.4f}, {tau1_sent:.4f}, {tau2_sent:.4f}) "
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
