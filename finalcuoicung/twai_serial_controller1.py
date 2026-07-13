import threading
import time
import math
import serial
import serial.tools.list_ports
import collections
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kinematic import get_acc_jerk
from trajectory import SplineTrajectory, QuinticTrajectory
from ctc_3dof import (CTC3Gains, CTC3Model, JointParams, LinkInertia3D,
                       ctc_3dof, ctc_3dof_components,
                       ctc_scalar_3dof, ctc_scalar_3dof_components,
                       PRISMATIC_MIN_MM, PRISMATIC_MAX_MM)

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
    "337535753034": "dev1",
    "3288365C3433": "dev2",
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
        debug_serial_verbose: bool = False,
    ):
        super().__init__(daemon=True)

        # ── Serial config ────────────────────────────────────────────────
        self.serial_port = serial_port
        self.baudrate    = baudrate
        self.ser: serial.Serial | None = None
        # Retry state cho permission-denied (USB đang reset, COM chưa sẵn sàng)
        self._connect_retry_delay_s = 2.0  # sleep giữa các lần thử kết nối

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
        self.start_pos = 0.0      
        self.motor_pos_rev = [0.0, 0.0, 0.0]   
        self.joint_pos_deg = [0.0, 0.0, 0.0]    
        self.joint_offset_deg = [0.0, 0.0, 0.0]  # thread FB: pos[i] = joint_raw - joint_offset_deg[i]
        self.q_max_deg = 0.0            # khớp knee: ROM (deg) — khởi tạo từ q_max_default
        self.q_max_default = 90.0      # ROM mặc định nếu PC không truyền (vd gõ "HOME" không tham số)
        self.model_home_deg = [0.0, 0.0, 0.0]
        # display_pos = pos + model_home_deg (chỉ phục vụ hiển thị Entry/plot real-time).
        # self.pos[] vẫn giữ hệ tương đối để setpoint/trajectory/velocity nhất quán.
        self.display_pos = [0.0, 0.0, 0.0]   # khởi tạo = 0, sẽ update khi nhận FB
        self.use_joint_feedback = True         # khi True, CTC dùng joint encoder làm feedback chính
        self.use_scalar_ctc = False            # True = dùng CTC đơn khớp (scalar Ic, không coupling M·C)
                                               #     False = dùng CTC đầy đủ ma trận M(q)·C(q,qdot) (mặc định)
                                               #     Scalar mode giống ODESC Trajectory_controller.dynamic_calculation(),
                                               #     chỉ phụ thuộc Izz + m·r² + gear²·J_motor của từng khâu độc lập.
                                               #     Khi model sai số lớn, scalar mode robust hơn.
        self.debug_sign_trace = False         # bật in [TWAI][TRACE] CTC/trajectory
        self.debug_serial_verbose = False     # True = in mọi dòng nhận từ ESP32 (tắt khi chạy ổn định)
        self.debug_ctc_log = False           # True = in CTC tau log mỗi 100Hz
        self._tx_print_count = 0
        self.debug_timing = False             # True = in thống kê timing vòng feedback mỗi 1s
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
        self.acc = [0.0, 0.0, 0.0]            # current acceleration (deg/s^2) — SavGol từ vel LP
        self.vel_set = [0.0, 0.0, 0.0]        # feedforward velocity (deg/s)
        self._last_pos_set = [0.0, 0.0, 0.0]
        self._last_set_ts = time.perf_counter()
        self._setpoint_dirty = False
        self.motion_armed = False
        self._motion_hold = False
        self._armed_last_pos = [0.0, 0.0, 0.0]
        self._armed_last_time = 0.0
        # Cache target đã gửi GOTO gần nhất để tránh spam khi ESP32 motion_active luôn = 1.
        self._last_sent_goto_targets = (None, None, None)
        self._last_goto_ts = 0.0
        self._home_pending = False
        self.target_tolerance_deg = 1.0
        self.target_tolerance_vel = 2.0
        self._last_motion_targets = [0.0, 0.0, 0.0] 
        # ── CTC plant + trajectory ──────────────────────────────────────
        # trajectory_mode: "spline" (mặc định) hoặc "quintic"
        #   - spline: cubic spline, mượt, có thể overshoot
        #   - quintic: đa thức bậc 5, đảm bảo v=0, a=0 ở 2 đầu → không gây giật khi Move
        # Đổi mode runtime bằng: self.ctrl.set_trajectory_mode("quintic")
        # GUI có combo Apply Trajectory để đổi + reset traj[] ngay.
        self.trajectory_mode = "spline"
        self.traj      = [SplineTrajectory(), SplineTrajectory(), SplineTrajectory()]
        self.acc_set   = [0.0, 0.0, 0.0]
        self.max_vel   = 60# °/s — giới hạn cho param_calc của spline (GUI có thể mở rộng sau)
        self._motion_t0           = -math.inf
        self._motion_time_active  = False
        self._was_motion_armed    = False
        self.tor_coef  = 1.0
        self.gear_ratio = gear_ratio_small
        # ── COM translation parameters (delta_x = prismatic translation along X) ──
        self.delta_x1 = 0.0   # Hip link COM translation (m)
        self.delta_x2 = 0.0   # Knee link COM translation (m)
        
        # Per-link physical parameters for the 3-DOF chain
        self.hip_link_com_distance_x1 = 0.128
        self.hip_link_com_distance_y1 = 0.02988
        self.hip_link_com_distance_z1 = 0.00191
        self.hip_link_com_distance_x2_base = 0.2838  # Base value before delta
        self.hip_link_com_distance_y2 = -0.0893
        self.hip_link_com_distance_z2 = 0.0
        self.knee_link_com_distance_x1 = 0.12843
        self.knee_link_com_distance_y1 = 0.03047
        self.knee_link_com_distance_z1 = 0.00162
        self.knee_link_com_distance_x2_base = 0.28359  # Base value before delta
        self.knee_link_com_distance_y2 = -0.0495
        self.knee_link_com_distance_z2 = -0.0428
        
        # ── Model 3-DOF CTC (initialized after _recalc_com_params) ────────
        self.ankle_link_com_distance = 0.05723
        
        # Backward-compatible aliases (MASS phải khai báo TRƯỚC khi tính inertia)
        self.hip_mass1 = self.hip_link_mass1 = 2.928
        self.hip_mass2 = self.hip_link_mass2 = 3.606
        self.knee_mass1 = self.knee_link_mass1 = 2.898
        self.knee_mass2 = self.knee_link_mass2 = 3.403
        self.ankle_mass = self.ankle_link_mass = 0.896
        # Aliases cho legacy functions
        self.hip_mass = self.hip_link_mass = self.hip_link_mass1 + self.hip_link_mass2
        self.knee_mass = self.knee_link_mass = self.knee_link_mass1 + self.knee_link_mass2
        self.link_mass = self.hip_mass + self.knee_mass + self.ankle_mass
        
        # Inertia parameters
        self.hip_link_inertia1 = 0.0214189039
        self.hip_link_inertia2 = 0.02863953781
        self.knee_link_inertia1 = 0.02099821589
        self.knee_link_inertia2 = 0.02698372936
        
        # COM distances từ joint (Euclidean) - cần tính TRƯỚC inertia
        hip_com_dist1 = math.sqrt(self.hip_link_com_distance_x1**2 + 
                                  self.hip_link_com_distance_y1**2 + 
                                  self.hip_link_com_distance_z1**2)
        hip_com_dist2 = math.sqrt(self.hip_link_com_distance_x2_base**2 + 
                                  self.hip_link_com_distance_y2**2 + 
                                  self.hip_link_com_distance_z2**2)
        knee_com_dist1 = math.sqrt(self.knee_link_com_distance_x1**2 + 
                                    self.knee_link_com_distance_y1**2 + 
                                    self.knee_link_com_distance_z1**2)
        knee_com_dist2 = math.sqrt(self.knee_link_com_distance_x2_base**2 + 
                                    self.knee_link_com_distance_y2**2 + 
                                    self.knee_link_com_distance_z2**2)
        
        # COM distance từ joint (weighted average - phải tính TRƯỚC khi dùng ở khâu parallel axis)
        self.hip_link_com_distance = (self.hip_link_mass1 * hip_com_dist1 +
                                       self.hip_link_mass2 * hip_com_dist2) / (self.hip_link_mass1 + self.hip_link_mass2)
        self.knee_link_com_distance = (self.knee_link_mass1 * knee_com_dist1 +
                                        self.knee_link_mass2 * knee_com_dist2) / (self.knee_link_mass1 + self.knee_link_mass2)

        # Khâu quán tính (parallel axis)
        self.hip_link_khau1 = (self.hip_link_mass1 * hip_com_dist1 +
                                self.hip_link_mass1 * self.hip_link_com_distance) / (self.hip_link_mass1 + self.hip_link_mass2)
        self.hip_link_khau2 = (self.hip_link_mass2 * hip_com_dist2 +
                                self.hip_link_mass2 * self.hip_link_com_distance) / (self.hip_link_mass1 + self.hip_link_mass2)
        self.knee_link_khau1 = (self.knee_link_mass1 * knee_com_dist1 +
                                 self.knee_link_mass1 * self.knee_link_com_distance) / (self.knee_link_mass1 + self.knee_link_mass2)
        self.knee_link_khau2 = (self.knee_link_mass2 * knee_com_dist2 +
                                 self.knee_link_mass2 * self.knee_link_com_distance) / (self.knee_link_mass1 + self.knee_link_mass2)
        
        # Inertia tổng
        self.hip_link_inertia = (self.hip_link_inertia1 + self.hip_link_inertia2 + 
                                  self.hip_link_mass1 * self.hip_link_khau1**2 + 
                                  self.hip_link_mass2 * self.hip_link_khau2**2)
        self.knee_link_inertia = (self.knee_link_inertia1 + self.knee_link_inertia2 + 
                                   self.knee_link_mass1 * self.knee_link_khau1**2 + 
                                   self.knee_link_mass2 * self.knee_link_khau2**2)
        self.small_motor_inertia = 0.000643
        self.big_motor_inertia = 0.002676
        self.hip_link_inertia_motor = self.big_motor_inertia * gear_ratio_big ** 2
        self.knee_link_inertia_motor = self.small_motor_inertia * gear_ratio_small ** 2
        self.ankle_inertia_motor = self.small_motor_inertia * gear_ratio_small ** 2
        self.ankle_link_inertia = 0.00274688527
        self.gear_ratios = (gear_ratio_big, gear_ratio_small, gear_ratio_small)
        
        # ── Prismatic joint parameters (từ VL53L0X) ─────────────────────────────
        # hip_link_length và knee_link_length thay đổi 350-450 mm theo prismatic
        self.prismatic_hip_mm = 350.0    # Vị trí prismatic hip (mm)
        self.prismatic_knee_mm = 350.0  # Vị trí prismatic knee (mm)
        self.hip_link_length = self.prismatic_hip_mm / 1000.0   # Chiều dài hip (m)
        self.knee_link_length = self.prismatic_knee_mm / 1000.0  # Chiều dài knee (m)
        self.ankle_link_length = 0.07    # Chiều dài ankle cố định (m)
        
        # Legacy aliases
        self.hip_distance = self.hip_link_com_distance
        self.knee_distance = self.knee_link_com_distance
        self.ankle_distance = self.ankle_link_com_distance
        self.gear_ratio = gear_ratio_small
        self.ext_load = 0.0
        self.hanger_mass = 0.0
        self.hanger_distance = 0.0
        self.coul_friction = 0.025
        self.visc_friction = 0.00276
        # ── Static friction (Stribeck) — Nm, chỉ phát huy khi |qdot| < threshold ──
        # 0.092 = |Trajectory_controller.py| gốc, dấu dương vì stFricDir = sign(qd - q)
        # trong ctc_3dof.py (khớp chiều). Tune lại bằng cách đo torque_min để motor
        # bắt đầu chuyển động.
        self.static_friction = 0.092
        self._recalc_com_params()
        self._recalc_plant_mass_inertia()

        # ── Vận tốc từ FB: dùng perf_counter + LP để tránh dt≈0 khi nhiều FB trong một lần đọc Serial
        self._fb_prev_pc: float | None = None
        self._fb_prev_p = [0.0, 0.0, 0.0]
        self._fb_have_prev = False
        self._vel_max_deg_s = 800.0
        # ── LP filter cho qdot (tốc độ phản hồi thực) ───────────────────────
        # fc: corner frequency (Hz). Lọc nhiễu encoder trước khi tính Kd·de.
        #   30-50 Hz: lọc mạnh, mượt, dùng với Kd cao (khuyến nghị mặc định)
        #   80-120 Hz: lọc nhẹ, phản ứng nhanh, dùng với Kd thấp
        #   công thức: alpha = 2π·fc / (2π·fc + fs), fs = 100 Hz (loop chính)
        # Mặc định 40 Hz để triệt nhiễu encoder AS5048A ở tốc độ cao.
        # (ESP32 vel LP 50Hz chỉ là 1-pole nhẹ, α≈0.758 → cần PC LP mạnh hơn
        # để giảm spike khi motor chạy nhanh + acceleration cao.)
        self._vel_lp_hz = 40.0
        self._vel_lp_alpha = _lp_alpha_from_fc(self._vel_lp_hz, 100.0)
        self._vel_lp_prev = [0.0, 0.0, 0.0]   # prev LP state
        # ── Median pre-filter (loại bỏ spike outlier trước khi LP) ─────────
        # raw vel_inst từ firmware = Δdeg / Δt có thể nhảy spike khi:
        #   - Encoder 1 LSB drift ở tốc độ cao (vd 0.022° / 0.01s = 2.2 deg/s)
        #   - dt_us bất thường nếu loop FB bị block (CAN pending, queue đầy)
        # Median 5-sample loại bỏ outlier rất hiệu quả mà giữ latency thấp (~25ms).
        self._vel_med_buf = [
            collections.deque(maxlen=5),
            collections.deque(maxlen=5),
            collections.deque(maxlen=5),
        ]
        # ── LP filter cho qdot_d (feedforward velocity từ trajectory) ─────────
        # Giữ riêng để có thể lọc trajectory mà không ảnh hưởng vel phản hồi.
        self._vel_set_lp_hz = 80.0
        self._vel_set_lp_alpha = _lp_alpha_from_fc(self._vel_set_lp_hz, 100.0)
        self._vel_set_lp_prev = [0.0, 0.0, 0.0]
        self._acc_set_lp_prev = [0.0, 0.0, 0.0]

        self.torque_set = [0.0, 0.0, 0.0]
        self._last_tau_raw = (0.0, 0.0, 0.0)
        self.use_torque_commands = True
        # Bridge mode: PC tính CTC và gửi torque/position xuống ESP32 forward tới ODrive.
        # ESP32 (main.c) KHÔNG tự tính CTC/trajectory - chỉ là pass-through bridge.
        self.on_board_ctl        = False  # PHẢI là False với firmware bridge hiện tại

        # ── Control params (dùng cho GUI control panel) ──────────────────
        self.Kp_axes = [3.0, 15.0, 3.0]
        self.Kd_axes = [1.0, 3.0, 1.0]   # Phải khớp với default trong guicontroller.py:_default_load
        # (Kp1=15, Kd1=3 cho M1 — bắt đầu thấp để tránh dao động;
        #  tăng dần khi đã verify motion ổn định)
        self.Kp3 = tuple(self.Kp_axes)
        self.Kd3 = tuple(self.Kd_axes)
        self.ctrl_bandwidth = 1000
        self.enc_bandwidth  = 100
        self.max_torque     = 0.9 # torque limit used for saturation in bridge mode
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
        self.motor_sign = (1, 1, 1)

        self.model3 = CTC3Model(
            joints=(
                JointParams(mass=self.hip_link_mass, length=self.hip_link_length, 
                           com_x=self.hip_link_chung_x, com_y=self.hip_link_chung_y,
                           inertia=self.hip_link_inertia,
                           motor_inertia=self.big_motor_inertia, gear_ratio=gear_ratio_big),
                JointParams(mass=self.knee_link_mass, length=self.knee_link_length, 
                           com_x=self.knee_link_chung_x, com_y=self.knee_link_chung_y,
                           inertia=self.knee_link_inertia,
                           motor_inertia=self.small_motor_inertia, gear_ratio=gear_ratio_small),
                JointParams(mass=self.ankle_link_mass, length=self.ankle_link_length, 
                           com_x=self.ankle_link_com_distance, com_y=0.0,
                           inertia=self.ankle_link_inertia,
                           motor_inertia=self.small_motor_inertia, gear_ratio=gear_ratio_small),
            ),
            gravity=g,
            # Static friction (Stribeck) — chỉ phát huy khi |qdot| < stribeck_vel_thresh.
            # Knee (j1) không có vì được dẫn động qua gear, ít ma sát tĩnh nhất.
            static_friction=(self.static_friction, 0.0, self.static_friction),
            coulomb_friction=(self.coul_friction, self.coul_friction, self.coul_friction),
            viscous_friction=(self.visc_friction, self.visc_friction, self.visc_friction),
            torque_scale=self.tor_coef,
            prismatic_hip_mm=self.prismatic_hip_mm,
            prismatic_knee_mm=self.prismatic_knee_mm,
        )

    # ════════════════════════════════════════════════════════════════════════
    # Connection
    # ════════════════════════════════════════════════════════════════════════

    def connect(self):
        """Mở cổng Serial. Gọi từ run() hoặc từ GUI thread.
        Nếu port bị chiếm (PermissionError/đang reset USB) → trả về False, run() loop sẽ retry.
        """
        # Đóng cổng cũ nếu từng mở nhưng đang lỗi
        if self.ser is not None:
            try:
                if self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None

        try:
            print(f"[TWAI] Kết nối tới {self.serial_port} @ {self.baudrate}...")
            self.ser = serial.Serial(
                self.serial_port,
                self.baudrate,
                timeout=0.02,
                write_timeout=0.5,   # tránh block vô hạn khi ESP32 busy
                rtscts=False,
                xonxoff=False,
            )

            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass
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
            # Gợi ý user nếu là lỗi permission
            if isinstance(e, PermissionError) or "Access is denied" in str(e):
                print(f"[TWAI] Tip: COM21 đang bị tiến trình khác giữ (Arduino Serial Monitor, ODrive Tool, instance cũ). "
                      f"Thoát hết rồi thử lại. Đợi {self._connect_retry_delay_s:.1f}s giữa các lần retry.")

    # === On-board mode helpers (ESP32 chạy CTC + trajectory) ===
    def goto(self, q0_deg: float, q1_deg: float, q2_deg: float):
        """Gửi lệnh GOTO tới ESP32 — ESP32 sẽ tính trajectory + CTC on-board."""
        self._send_simple_cmd(f"GOTO {q0_deg:.4f} {q1_deg:.4f} {q2_deg:.4f}")

    def hold(self):
        """Dừng motion trên ESP32, torque về 0."""
        self._send_simple_cmd("HOLD")

    def set_gains(self, kp0: float, kp1: float, kp2: float,
                  kd0: float, kd1: float, kd2: float):
        """Gửi GAIN tới ESP32 — thay đổi Kp/Kd on-board runtime."""
        self._send_simple_cmd(f"GAIN {kp0:.4f} {kp1:.4f} {kp2:.4f} "
                              f"{kd0:.4f} {kd1:.4f} {kd2:.4f}")

    def set_max_vel(self, vmax_deg_s: float):
        """Đặt max velocity cho SplineTrajectory trên ESP32."""
        self._send_simple_cmd(f"VMAX {vmax_deg_s:.4f}")

    def set_prismatic(self, hip_mm: float, knee_mm: float):
        """Gửi prismatic positions tới ESP32 (mm).
        
        hip_mm: Vị trí prismatic hip (350-450 mm)
        knee_mm: Vị trí prismatic knee (350-450 mm)
        """
        # Clamp to valid range
        hip_mm = max(PRISMATIC_MIN_MM, min(PRISMATIC_MAX_MM, hip_mm))
        knee_mm = max(PRISMATIC_MIN_MM, min(PRISMATIC_MAX_MM, knee_mm))
        
        # Update local model
        self.prismatic_hip_mm = hip_mm
        self.prismatic_knee_mm = knee_mm
        self.hip_link_length = hip_mm / 1000.0
        self.knee_link_length = knee_mm / 1000.0
        
        # Update CTC model
        self.model3.prismatic_hip_mm = hip_mm
        self.model3.prismatic_knee_mm = knee_mm
        self.model3.update_prismatic_lengths()
        
        # Send to ESP32
        self._send_simple_cmd(f"PRISM {hip_mm:.1f} {knee_mm:.1f}")

    def control_cylinder(self, joint_idx: int, direction: int):
        """Điều khiển cylinder: joint_idx=0(hip),1(knee), direction=1(kéo dài),-1(thu ngắn)."""
        # joint_idx: 0=hip, 1=knee
        # direction: 1=kéo dài (+), -1=thu ngắn (-)
        cmd = f"CYL {joint_idx} {direction}"
        self._send_simple_cmd(cmd)

    def set_locked_axes(self, lock0: bool, lock1: bool, lock2: bool):
        """Gửi lệnh LOCK tới ESP32 — khóa axes không nhận torque.
        
        lock0, lock1, lock2: True = khóa axis (torque = 0)
        """
        self._send_simple_cmd(f"LOCK {int(lock0)} {int(lock1)} {int(lock2)}")

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
            # Cổng đang lỗi → đánh dấu disconnect, run() loop sẽ reconnect
            self.connected = False
            # Đừng spam flood mỗi 10ms — chờ 0.5s
            time.sleep(0.5)

    def enter_closed_loop(self, clear_errors: bool = True):
        """Kích hoạt chế độ torque/position nhưng chưa chạy motion.

        Args:
            clear_errors: True (mặc định) → gửi CLEAR errors trước khi vào closed loop.
                          False → giữ nguyên state errors hiện tại (chỉ dùng khi đã rõ).
        """
        if not self.esp32_ready:
            self.pending_closed_loop = True
            self.status_message = "Đang chờ ESP32 READY để vào Closed Loop"
            print("[TWAI] ESP32 chưa READY, sẽ tự vào Closed Loop khi READY.")
            return
        self.pending_closed_loop = False
        self.motion_armed = False
        self._setpoint_dirty = False
        if clear_errors:
            self._clear_odrive_errors(why="enter_closed_loop")
        else:
            # Backward-compat: vẫn gửi CLEAR thủ công (giống hành vi cũ).
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

    def set_delta_x(self, delta_x1: float, delta_x2: float):
        """Cập nhật khoảng tịnh tiến COM và tính lại CTC model.
        
        Args:
            delta_x1: Khoảng tịnh tiến hip COM theo X (m)
            delta_x2: Khoảng tịnh tiến knee COM theo X (m)
        """
        self.delta_x1 = float(delta_x1)
        self.delta_x2 = float(delta_x2)
        self._recalc_com_params()
        print(f"[TWAI] Delta_x updated: delta_x1={self.delta_x1:.4f}m, delta_x2={self.delta_x2:.4f}m")

    def emergency_stop(self):
        self.pending_closed_loop = False
        self._estop_event.set()
        self._reset_torque_slew()
        self.motion_armed = False
        self._was_motion_armed = False
        self._motion_time_active = False
        self._vel_lp_prev = [0.0, 0.0, 0.0]
        self._vel_set_lp_prev = [0.0, 0.0, 0.0]
        self._acc_set_lp_prev = [0.0, 0.0, 0.0]
        for buf in getattr(self, "_vel_med_buf", []):
            buf.clear()
        if self.on_board_ctl:
            self.hold()
        else:
            self._send_simple_cmd("IDLE")
        self.status_message = "ESTOP!"
        print("[TWAI] EMERGENCY STOP!")

    def reset(self, clear_errors: bool = True):
        """Reset state flags.

        Args:
            clear_errors: True (mặc định) → gửi CLEAR errors xuống ODrive.
                          False → chỉ reset state flags (legacy).
        """
        self._estop_event.clear()
        self.pending_closed_loop = False
        self.isOffset = False
        self.motion_armed = False
        self._motion_hold = False
        self._home_pending = False
        self._new_motion_pending = False       # set bởi GUI khi nhấn Run → restart clock
        self._setpoint_dirty = False
        self._motion_time_active = False
        self._motion_t0 = -math.inf
        self._was_motion_armed = False
        # Reset lock về False để ESP32 nhận torque bình thường trở lại
        self.locked_axes = [False, False, False]
        self.set_locked_axes(False, False, False)
        for traj in self.traj:
            traj.reset()
        with self.data_lock:
            self._fb_have_prev = False
            self._fb_prev_pc = None
            self.vel[0] = 0.0
            self.vel[1] = 0.0
            self.vel[2] = 0.0
            self.acc[0] = 0.0
            self.acc[1] = 0.0
            self.acc[2] = 0.0
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
        if clear_errors:
            self._clear_odrive_errors(why="reset")
        else:
            self._send_simple_cmd("CLEAR")
        self.status_message = "Reset xong"
        print("[TWAI] Reset.")

    def stop(self):
        self._stop_event.set()
        self.disconnect()

    # ════════════════════════════════════════════════════════════════════════
    # Offset (lấy vị trí hiện tại làm gốc)
    # ════════════════════════════════════════════════════════════════════════

    def set_offset(self, clear_errors: bool = True):
        """
        Lưu vị trí hiện tại làm home.

        Logic cũ (anh yêu cầu):
        - Sau khi bấm Offset, Pos Motor 1 (display) = -90° world.
        - Tính toán CTC dùng q = -90° (chứ không phải 0°).
        - Sau KOK home (limit switch): display = 0° world (vị trí mới).

        Cách làm:
        - Đặt pos[1] = -90° (giá trị world âm).
        - Đặt joint_offset_deg[1] = -raw_joint1 - (-90) = raw_joint1 - pos[1]_cũ
          để thread FB tiếp tục tính: pos[1] = raw - offset_raw = -90°.
          Thực ra đơn giản hơn: pos[1] = -90, model_home_deg[1] = 0,
          joint_offset_deg[1] = -raw_at_set_offset + (-90).
          Nhưng raw có thể dao động. Cách an toàn nhất:
          pos[1] = -90.0 (giá trị world cố định).
          joint_offset_deg[1] = -raw_joint1 + (-90) ban đầu, nhưng sau đó thread
          sẽ ghi lại pos[1] = raw - offset → dao động quanh -90.

        → Cách TỐT NHẤT: set pos[1] = -90 và đặt cờ set_offset_just_done để thread FB
        KHÔNG ghi đè pos[1] trong ~200ms sau khi bấm Offset. Sau 200ms, raw FB sẽ được
        chấp nhận (lúc đó user mong đợi pos[1] = raw - offset, nên chọn offset sao cho
        raw - offset = -90).
        """
        if clear_errors:
            self._clear_odrive_errors(why="set_offset")
        with self.data_lock:
            self.offset_rev[0] = self.motor_pos_rev[0]
            self.offset_rev[1] = self.motor_pos_rev[1]
            self.offset_rev[2] = self.motor_pos_rev[2]

            # Lưu raw joint[1] TRƯỚC khi reset để tính offset_raw sao cho
            # raw - offset_raw = -90.
            raw_joint_1 = self.joint_pos_deg[1] if self.use_joint_feedback else None

            for i in range(3):
                self.joint_pos_deg[i] = 0.0
                self.pos[i] = 0.0
                self.pos_set[i] = 0.0
                self.vel_set[i] = 0.0
                self.acc_set[i] = 0.0
                self._last_pos_set[i] = 0.0
                self._last_motion_targets[i] = 0.0

            # LOGIC CŨ NGÀY XƯA (anh yêu cầu):
            # pos[1] = -90° (world), model_home_deg[1] = 0 → display = -90°.
            # joint_offset_deg[1] = raw - (-90) = raw + 90 → thread FB: pos[1] = raw - (raw+90) = -90°.
            self.pos[1] = -90.0
            self.model_home_deg[1] = 0.0
            if self.use_joint_feedback and raw_joint_1 is not None:
                self.joint_offset_deg[1] = raw_joint_1 + 90.0
            else:
                self.joint_offset_deg[1] = 0.0
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
            self.acc[0] = self.acc[1] = self.acc[2] = 0.0
            self._tau_out[0] = self._tau_out[1] = self._tau_out[2] = 0.0
            self._clear_vel_fit_buffers_locked()
            self._timing_dts = []
            self._timing_latency = []
            self._timing_last_report = 0.0
            self._timing_last_pc_s = 0.0
            for traj in self.traj:
                traj.reset()
            # Reset data buffer để plot/snapshot bắt đầu tươi từ vị trí Set Offset
            # (= model_home_deg cho Entry/plot real-time).
            self.data.clear()
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
        # Cũng reset median pre-filter.
        for buf in getattr(self, "_vel_med_buf", []):
            buf.clear()

    def _update_vel_estimates_locked(self, t_wall: float, v0_inst: float, v1_inst: float, v2_inst: float) -> None:
        """Median → LP → SavGol pipeline để triệt nhiễu encoder trước khi tính acc/Kd·vel.

        Pipeline (mỗi axis):
          v_inst (raw từ ESP32) → median 5 → 1-pole LP 40Hz → SavGol(w,p)
        Median loại bỏ outlier spikes (1 LSB drift ở tốc độ cao, dt jitter khi loop block).
        LP 40Hz mượt phần còn lại. SavGol tính acc trên vel đã sạch.
        """
        n_axes = 3
        v_insts = (v0_inst, v1_inst, v2_inst)
        # Ghi vào từng buffer theo từng joint; đồng thời cập nhật vel (LP-filter).
        for i, v_inst in enumerate(v_insts):
            # ── Median pre-filter (loại bỏ outlier spikes) ──
            med_buf = self._vel_med_buf[i]
            med_buf.append(v_inst)
            if len(med_buf) == med_buf.maxlen:
                # Median của 5 sample (đã sort tăng dần → lấy phần tử giữa).
                sorted_vals = sorted(med_buf)
                v_med = sorted_vals[len(sorted_vals) // 2]
            else:
                v_med = v_inst  # Buffer chưa đầy → dùng raw (chỉ vài tick đầu).
            self.velFilBuf[i].append(v_med)
            self.timeFilBuf[i].append(t_wall)
            if len(self.velFilBuf[i]) == self.window_size:
                try:
                    vf, acc, _ = get_acc_jerk(
                        np.asarray(self.timeFilBuf[i], dtype=float),
                        np.asarray(self.velFilBuf[i], dtype=float),
                        self.window_size,
                        self.poly_order,
                    )
                    lim = self._vel_max_deg_s
                    vf_clipped = max(min(float(vf), lim), -lim)
                except (ValueError, IndexError, np.linalg.LinAlgError):
                    # SavGol fail khi buffer quá nhỏ/NaN/singular — fallback về raw vel,
                    # acc = 0. KHÔNG bubble exception ra ngoài (trước đây làm hỏng FB).
                    vf_clipped = v_inst
                    acc = 0.0
                a = self._vel_lp_alpha
                self.vel[i] = a * vf_clipped + (1.0 - a) * self._vel_lp_prev[i]
                self._vel_lp_prev[i] = self.vel[i]
                # Acc pre-computed (deg/s^2) — GUI chỉ đọc, không tính lại.
                self.acc[i] = float(acc)
            else:
                # Chưa đủ sample cho SavGol: dùng LP thẳng trên v_inst, acc = 0.
                a = self._vel_lp_alpha
                self.vel[i] = a * v_inst + (1.0 - a) * self._vel_lp_prev[i]
                self._vel_lp_prev[i] = self.vel[i]
                self.acc[i] = 0.0

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

    def _recalc_com_params(self):
        """Tính lại COM parameters khi delta_x1 hoặc delta_x2 thay đổi.
        
        delta_x1: khoảng tịnh tiến COM hip link theo X (m)
        delta_x2: khoảng tịnh tiến COM knee link theo X (m)
        """
        # COM x2 với delta translation
        hip_link_com_distance_x2 = self.hip_link_com_distance_x2_base + self.delta_x1
        knee_link_com_distance_x2 = self.knee_link_com_distance_x2_base + self.delta_x2
        
        # Matrix COM distance (weighted average theo khối lượng)
        self.hip_link_chung_x = (self.hip_mass1 * self.hip_link_com_distance_x1 + 
                                  self.hip_mass2 * hip_link_com_distance_x2) / (self.hip_mass1 + self.hip_mass2)
        self.hip_link_chung_y = (self.hip_mass1 * self.hip_link_com_distance_y1 + 
                                  self.hip_mass2 * self.hip_link_com_distance_y2) / (self.hip_mass1 + self.hip_mass2)
        self.hip_link_chung_z = (self.hip_mass1 * self.hip_link_com_distance_z1 + 
                                  self.hip_mass2 * self.hip_link_com_distance_z2) / (self.hip_mass1 + self.hip_mass2)
        
        self.knee_link_chung_x = (self.knee_mass1 * self.knee_link_com_distance_x1 + 
                                   self.knee_mass2 * knee_link_com_distance_x2) / (self.knee_mass1 + self.knee_mass2)
        self.knee_link_chung_y = (self.knee_mass1 * self.knee_link_com_distance_y1 + 
                                   self.knee_mass2 * self.knee_link_com_distance_y2) / (self.knee_mass1 + self.knee_mass2)
        self.knee_link_chung_z = (self.knee_mass1 * self.knee_link_com_distance_z1 + 
                                   self.knee_mass2 * self.knee_link_com_distance_z2) / (self.knee_mass1 + self.knee_mass2)
        
        # COM distance từ joint (Euclidean)
        hip_com_dist1 = math.sqrt(self.hip_link_com_distance_x1**2 + 
                                  self.hip_link_com_distance_y1**2 + 
                                  self.hip_link_com_distance_z1**2)
        hip_com_dist2 = math.sqrt(hip_link_com_distance_x2**2 + 
                                  self.hip_link_com_distance_y2**2 + 
                                  self.hip_link_com_distance_z2**2)
        knee_com_dist1 = math.sqrt(self.knee_link_com_distance_x1**2 + 
                                    self.knee_link_com_distance_y1**2 + 
                                    self.knee_link_com_distance_z1**2)
        knee_com_dist2 = math.sqrt(knee_link_com_distance_x2**2 + 
                                    self.knee_link_com_distance_y2**2 + 
                                    self.knee_link_com_distance_z2**2)
        
        # Weighted COM distance
        self.hip_link_com_distance = (self.hip_link_mass1 * hip_com_dist1 + 
                                       self.hip_link_mass2 * hip_com_dist2) / (self.hip_link_mass1 + self.hip_link_mass2)
        self.knee_link_com_distance = (self.knee_link_mass1 * knee_com_dist1 + 
                                        self.knee_link_mass2 * knee_com_dist2) / (self.knee_link_mass1 + self.knee_link_mass2)
        
        # Khâu quán tính (parallel axis theorem)
        self.hip_link_khau1 = (self.hip_link_mass1 * hip_com_dist1 + 
                                (self.hip_link_mass1 + self.hip_link_mass2) * self.hip_link_com_distance) / (2 * self.hip_link_mass1 + self.hip_link_mass2)
        self.hip_link_khau2 = (self.hip_link_mass2 * hip_com_dist2 + 
                                (self.hip_link_mass1 + self.hip_link_mass2) * self.hip_link_com_distance) / (self.hip_link_mass1 + 2 * self.hip_link_mass2)
        self.knee_link_khau1 = (self.knee_link_mass1 * knee_com_dist1 + 
                                 (self.knee_link_mass1 + self.knee_link_mass2) * self.knee_link_com_distance) / (2 * self.knee_link_mass1 + self.knee_link_mass2)
        self.knee_link_khau2 = (self.knee_link_mass2 * knee_com_dist2 + 
                                 (self.knee_link_mass1 + self.knee_link_mass2) * self.knee_link_com_distance) / (self.knee_link_mass1 + 2 * self.knee_link_mass2)
        
        # Inertia tổng
        self.hip_link_inertia = (self.hip_link_inertia1 + self.hip_link_inertia2 + 
                                  self.hip_link_mass1 * self.hip_link_khau1**2 + 
                                  self.hip_link_mass2 * self.hip_link_khau2**2)
        self.knee_link_inertia = (self.knee_link_inertia1 + self.knee_link_inertia2 + 
                                   self.knee_link_mass1 * self.knee_link_khau1**2 + 
                                   self.knee_link_mass2 * self.knee_link_khau2**2)
        
        # Cập nhật CTC model nếu đã được khởi tạo
        if hasattr(self, 'model3') and self.model3 is not None:
            self.model3.joints[0].com_x = self.hip_link_chung_x
            self.model3.joints[0].com_y = self.hip_link_chung_y
            self.model3.joints[1].com_x = self.knee_link_chung_x
            self.model3.joints[1].com_y = self.knee_link_chung_y
            self.model3.joints[0].inertia = LinkInertia3D.from_scalar(self.hip_link_inertia, self.hip_link_mass, self.hip_link_chung_x, self.hip_link_chung_y)
            self.model3.joints[1].inertia = LinkInertia3D.from_scalar(self.knee_link_inertia, self.knee_link_mass, self.knee_link_chung_x, self.knee_link_chung_y)
        
        print(f"[TWAI] COM updated: hip_chung=({self.hip_link_chung_x:.4f}, {self.hip_link_chung_y:.4f}), "
              f"knee_chung=({self.knee_link_chung_x:.4f}, {self.knee_link_chung_y:.4f})")

    def _init_trajectories(self):
        """Khởi tạo lại 3 trajectory object theo self.trajectory_mode.
        Chỉ gọi trong data_lock. Dùng khi đổi mode lúc runtime hoặc reset."""
        from trajectory import (SplineTrajectory, QuinticTrajectory,
                                CubicTrajectory, TrapezoidalTrajectory)
        cls_map = {
            "quintic": QuinticTrajectory,
            "cubic": CubicTrajectory,
            "trapezoidal": TrapezoidalTrajectory,
        }
        traj_cls = cls_map.get(self.trajectory_mode, SplineTrajectory)
        self.traj = [traj_cls(), traj_cls(), traj_cls()]
        self._motion_time_active = False
        self._motion_t0 = -math.inf
        self.pos_set = [0.0, 0.0, 0.0]
        self.vel_set = [0.0, 0.0, 0.0]
        self.acc_set = [0.0, 0.0, 0.0]

    def set_trajectory_mode(self, mode: str):
        """Đổi mode trajectory từ runtime ('spline' | 'quintic' | 'cubic' | 'trapezoidal').
        Có thể gọi không cần data_lock (hàm tự lấy)."""
        mode = (mode or "spline").strip().lower()
        valid_modes = ("spline", "quintic", "cubic", "trapezoidal")
        if mode not in valid_modes:
            print(f"[TWAI] set_trajectory_mode: invalid mode '{mode}', keeping '{self.trajectory_mode}'")
            return
        self.trajectory_mode = mode
        with self.data_lock:
            self._init_trajectories()
        print(f"[TWAI] Trajectory mode set to '{mode}', traj[] re-initialized")

    def _make_trajectory(self):
        from trajectory import (SplineTrajectory, QuinticTrajectory,
                                CubicTrajectory, TrapezoidalTrajectory)
        cls_map = {
            "quintic": QuinticTrajectory,
            "cubic": CubicTrajectory,
            "trapezoidal": TrapezoidalTrajectory,
        }
        traj_cls = cls_map.get(self.trajectory_mode, SplineTrajectory)
        return traj_cls()

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
        self._acc_set_lp_prev = [0.0, 0.0, 0.0]
        for buf in self._vel_med_buf:
            buf.clear()

    def _start_new_motion(self):
        """Restart motion clock + reset hold state (ODESC-style).

        GUI gọi mỗi lần user nhấn Run với target mới — tương đương ODESC
        update_ctrlElms() set self.t_ref = time.time(). Sau move xong, motion_armed
        vẫn True (không disarm), CTC giữ pos nhờ G_hold blend. Khi target mới
        đến, method này restart _motion_t0 để spline chạy lại từ đầu.
        """
        with self.data_lock:
            self._motion_hold = False
            # ODESC-style: arm motion ngay tại đây. update_ctrlElms() của GUI
            # set motion_armed=False, nhưng Run phải re-arm để CTC chạy mỗi tick.
            self.motion_armed = True
            self._new_motion_pending = True     # flag cho run-loop restart clock
            # Reset acc_set LP filter để tránh spike từ giá trị cũ (acceleration cuối motion).
            self._acc_set_lp_prev = [0.0, 0.0, 0.0]      

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
        qddot_d = []
        for i in range(3):
            as_raw = self.acc_set[i] * DEG2RAD
            a = self._vel_set_lp_alpha  # dùng cùng hệ số LP cho feedforward
            as_filt = a * as_raw + (1.0 - a) * self._acc_set_lp_prev[i]
            self._acc_set_lp_prev[i] = as_filt
            qddot_d.append(as_filt)

        # Smooth-startup: t_prog tính từ lúc bắt đầu motion để blend G_hold→G_ff.
        if self._motion_time_active:
            startup_t = max(time.time() - self._motion_t0, 0.0)
        else:
            startup_t = 0.0

        # Chọn CTC mode: scalar (đơn khớp, giống ODESC) hoặc full (M·C·G coupling).
        if self.use_scalar_ctc:
            comp = ctc_scalar_3dof_components(qd, q, qdot_d, qdot, qddot_d, gains, self.model3, startup_t=startup_t)
        else:
            comp = ctc_3dof_components(qd, q, qdot_d, qdot, qddot_d, gains, self.model3, startup_t=startup_t)
        tau = comp["tau"]
        self._last_tau_raw = tuple(tau)
        for i in range(3):
            if active_axis is None or i == active_axis:
                self.torque_set[i] = float(tau[i])
            else:
                self.torque_set[i] = 0.0
        # ── Debug CTC heartbeat (mỗi 2s, chỉ khi motion_armed) ──
        # Giúp chẩn đoán tau = 0 hoặc dao động mà không flood log.
        if self.motion_armed and not hasattr(self, "_ctc_dbg_last_ts"):
            self._ctc_dbg_last_ts = 0.0
        if self.motion_armed and (time.time() - getattr(self, "_ctc_dbg_last_ts", 0.0)) >= 2.0:
            self._ctc_dbg_last_ts = time.time()
            mode_tag = "SCALAR" if self.use_scalar_ctc else "FULL"
            # In 1 dòng tóm tắt tau + q/qd/e để chẩn đoán nhanh.
            # e ở deg, q1 đang dao động nhiều → e lớn → tau lớn → Kp3 quá cao?
            e1 = math.degrees(q[1] - qd[1])
            print(
                f"[TWAI][CTC] mode={mode_tag} "
                f"tau=({tau[0]:+.3f},{tau[1]:+.3f},{tau[2]:+.3f}) "
                f"qd=({math.degrees(qd[0]):+.2f},{math.degrees(qd[1]):+.2f},{math.degrees(qd[2]):+.2f}) "
                f"q1={math.degrees(q[1]):+.2f} e1={e1:+.2f}deg "
                f"Kp3=({self.Kp3[0]:.1f},{self.Kp3[1]:.1f},{self.Kp3[2]:.1f}) "
                f"Kd3=({self.Kd3[0]:.3f},{self.Kd3[1]:.3f},{self.Kd3[2]:.3f})"
            )
        if trace:
            G = comp["g"]
            p_term = comp["p_term"]
            d_term = comp["d_term"]
            w = comp["startup_w"]
            mode_tag = "SCALAR" if self.use_scalar_ctc else "FULL"
            # Log đầy đủ các thành phần CTC để debug steady-state error / lag.
            # FULL mode có mv/cv; SCALAR mode có m_i (scalar inertia).
            extra = ""
            if self.use_scalar_ctc:
                mi = comp.get("m_i", (0.0, 0.0, 0.0))
                extra = f" I=({mi[0]:.3f},{mi[1]:.3f},{mi[2]:.3f})"
            else:
                mv = comp.get("mv", (0.0, 0.0, 0.0))
                cv = comp.get("cv", (0.0, 0.0, 0.0))
                extra = f" Mv=({mv[0]:.2f},{mv[1]:.2f},{mv[2]:.2f}) Cv=({cv[0]:.2f},{cv[1]:.2f},{cv[2]:.2f})"
            print(
                f"[TWAI][TRACE] ctc mode={mode_tag} t_prog={startup_t:.3f}s w={w:.2f} "
                f"q_deg=({q_source[0]:.2f},{q_source[1]:.2f},{q_source[2]:.2f}) "
                f"qd_deg=({self.pos_set[0]:.2f},{self.pos_set[1]:.2f},{self.pos_set[2]:.2f}) "
                f"e=({q[0]-qd[0]:.3f},{q[1]-qd[1]:.3f},{q[2]-qd[2]:.3f}) "
                f"PD=({p_term[0]:.2f},{p_term[1]:.2f},{p_term[2]:.2f}) + "
                f"({d_term[0]:.2f},{d_term[1]:.2f},{d_term[2]:.2f}) "
                f"G=({G[0]:.2f},{G[1]:.2f},{G[2]:.2f})"
                f"{extra} "
                f"tau_raw=({self.torque_set[0]:.2f},{self.torque_set[1]:.2f},{self.torque_set[2]:.2f})"
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
            # Rate-limit TX prints to avoid flooding console at 100Hz
            if self.debug_serial_verbose and self._tx_print_count % 50 == 0:
                print(f"[TWAI] SER TX -> \"{cmd.strip()}\"")
            self._tx_print_count += 1
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
                # ser.read(n) trả về tối đa n bytes — có thể < n nếu timeout
                # hoặc race với thread đọc khác. Dùng min() để tránh IndexError nội bộ.
                n_to_read = min(int(waiting), 4096)
                raw = self.ser.read(n_to_read).decode("ascii", errors="replace")
                self._line_buf += raw

                # Parse từng dòng trong try/except riêng — 1 dòng lỗi KHÔNG được
                # ngắt kết nối (trước đây IndexError ở parse làm self.connected=False).
                while "\n" in self._line_buf:
                    line, self._line_buf = self._line_buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if self.debug_serial_verbose:
                        print(f"[TWAI] RX raw: '{line}'")
                    try:
                        self._parse_line(line)
                    except IndexError as e:
                        # IndexError rất cụ thể ("deque index out of range") — cần traceback
                        # để biết deque nào và dòng nào. In full traceback để debug.
                        import traceback as _tb
                        print(f"[TWAI] Parse IndexError (skip line): {e} — line='{line[:80]}'")
                        _tb.print_exc()
                    except ValueError as e:
                        # Firmware gửi số không parse được (NaN, inf, garbage).
                        print(f"[TWAI] Parse ValueError (skip line): {e} — line='{line[:80]}'")
                    except Exception as e:
                        # Lỗi khác (vd race condition khác) — log trace, skip.
                        import traceback as _tb
                        print(f"[TWAI] Parse error (skip line): {type(e).__name__}: {e}")
                        _tb.print_exc()
        except Exception as e:
            # Lỗi thực sự từ tầng serial (cable rút, port đóng, OS reset).
            # CHỈ trường hợp này mới set connected=False.
            import traceback as _tb
            print(f"[TWAI] Lỗi đọc Serial: {e}")
            _tb.print_exc()
            self.connected = False

    def _parse_line(self, line: str):
        """Xử lý một dòng text từ ESP32.

        Firmware ESP32 (main.c bridge mode) gửi format CSV:
          FB,mot0_rev,mot1_rev,mot2_rev,j0_deg,j1_deg,j2_deg,tau0_Nm,tau1_Nm,tau2_Nm,motion_active
        Index: 0         1         2         3       4       5       6        7        8        9
        motion_active: 1 = đang chạy (sau GOTO), 0 = HOLD/idle
        """
        if self.debug_serial_verbose:
            print(f"[TWAI] <<< {line}")
        if line.startswith("FB,"):
            parts = line[3:].split(",")
            # Format: FB,mot0,mot1,mot2,joint0,joint1,joint2,tau0,tau1,tau2,motion[,vel0,vel1,vel2]
            # Index:      0     1     2     3      4      5      6     7     8     9   [10..12]
            if len(parts) < 10:
                print(f"[TWAI] FB format error: {len(parts)} values — '{line}'")
                return
            try:
                raw0  = float(parts[0])  # motor pos0 (revs)
                raw1  = float(parts[1])  # motor pos1
                raw2  = float(parts[2])  # motor pos2
                j0    = float(parts[3])  # joint_deg[0]
                j1    = float(parts[4])  # joint_deg[1]
                j2    = float(parts[5])  # joint_deg[2]
                # Parse torque (firmware luôn gửi float hợp lệ, không 'nan')
                tau0 = float(parts[6])
                tau1 = float(parts[7])
                tau2 = float(parts[8])
                # motion_active flag
                motion_active_int = int(parts[9]) if parts[9].isdigit() else 0
                # Velocity từ ESP32 (deg/s) - thay vì Python tự tính
                vel_esp0 = float(parts[10]) if len(parts) > 10 else 0.0
                vel_esp1 = float(parts[11]) if len(parts) > 11 else 0.0
                vel_esp2 = float(parts[12]) if len(parts) > 12 else 0.0
                if self.debug_serial_verbose:
                    print(f"[TWAI] RAW joint=({j0:.3f},{j1:.3f},{j2:.3f}) tau=({tau0:.4f},{tau1:.4f},{tau2:.4f}) motion={motion_active_int} vel=({vel_esp0:.3f},{vel_esp1:.3f},{vel_esp2:.3f})")
                # Timestamp: firmware không gửi, dùng PC time (không tính latency).
                # Đã tắt debug_timing theo mặc định; nếu cần latency thật, thêm timestamp vào firmware.
                ts_ms = 0
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

                # Motor encoder in degrees luôn được tính sẵn từ _rev_to_deg ở trên.
                # Tuy nhiên, KHÔNG đưa vào data tuple nữa để giảm overhead — GUI không cần.
                # Nếu cần debug, mở _parse_line / log riêng.

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

                # display_pos: cộng model_home_deg để Entry/plot real-time hiển thị giá trị world.
                # Chỉ phục vụ GUI — KHÔNG dùng cho setpoint, trajectory hay velocity.
                self.display_pos[0] = self.pos[0] + self.model_home_deg[0]
                self.display_pos[1] = self.pos[1] + self.model_home_deg[1]
                self.display_pos[2] = self.pos[2] + self.model_home_deg[2]  
                # Velocity from ESP32 (deg/s) - firmware tính từ joint_deg delta / 0.01s
                lim = self._vel_max_deg_s
                self._update_vel_estimates_locked(
                    now,
                    max(min(vel_esp0, lim), -lim),
                    max(min(vel_esp1, lim), -lim),
                    max(min(vel_esp2, lim), -lim),
                )
                # Vẫn track prev để compute accel bằng get_acc_jerk
                self._fb_prev_pc = now_pc
                self._fb_prev_p[0] = self.pos[0]
                self._fb_prev_p[1] = self.pos[1]
                self._fb_prev_p[2] = self.pos[2]
                self._fb_have_prev = True

                # Compute CTC tau ngay khi nhận FB để GUI plot có giá trị raw CTC.
                # Chỉ tính khi closed_loop + đã set offset (CTC cần q thật + setpoint).
                if self.closed_loop_control and self.isOffset:
                    try:
                        self._refresh_traj_refs_locked(now)
                        self._dynamic_calculation_locked(active_axis=None, trace=self.debug_sign_trace)
                    except Exception:
                        # Không để lỗi CTC làm hỏng FB handler, NHƯNG phải log để debug.
                        # Nếu nuốt hoàn toàn → tau = 0 suốt motion → motor overshoot vì không có torque kéo về.
                        if not hasattr(self, "_ctc_err_count"):
                            self._ctc_err_count = 0
                            self._ctc_err_last_log_ts = 0.0
                        self._ctc_err_count += 1
                        if time.time() - self._ctc_err_last_log_ts >= 1.0:
                            self._ctc_err_last_log_ts = time.time()
                            import traceback as _tb
                            print(f"[TWAI][CTC] calc failed x{self._ctc_err_count}:")
                            _tb.print_exc()

            # Use torque from ESP32 (last sent). Lưu ý: firmware luôn gửi giá trị
            # torque mới nhất mỗi frame FB, kể cả = 0 (để biết "không lệnh").
            # Trước đây check "if tau != 0" gây sticky value khi motor đã dừng.
            self._tau_out[0] = tau0
            self._tau_out[1] = tau1
            self._tau_out[2] = tau2
            # Debug print every 500ms
            if not hasattr(self, '_last_tau_print'):
                self._last_tau_print = 0.0
            if self.debug_ctc_log and now - self._last_tau_print >= 0.5:
                self._last_tau_print = now
                print(f"[CTC] tau=({tau0:.4f},{tau1:.4f},{tau2:.4f}) q=({j0:.2f},{j1:.2f},{j2:.2f}) motion={motion_active_int}")

            # motion_armed giờ do GUI Move button set (rising edge từ _on_move)
            # và CHỈ tắt khi user nhấn IDLE/ESTOP/Reset. ESP32 báo motion_active_int
            # không tự động disarm PC (tránh race condition khi firmware chưa bật
            # motion_active ngay sau khi nhận torque đầu tiên).
            # Log motion_active để debug.
            if not hasattr(self, "_last_motion_active_log_ts"):
                self._last_motion_active_log_ts = 0.0
            if now - self._last_motion_active_log_ts >= 1.0:
                self._last_motion_active_log_ts = now
                print(f"[TWAI][MOTION_FB] motion_active_int={motion_active_int} armed={self.motion_armed}")

            # Cấu trúc data tuple (22 field — ODESC-style, đủ cho plot & log):
            #   0      : timestamp (float, seconds)
            #   1-3    : pos[0..2]         (deg, joint-side)
            #   4-6    : pos_set[0..2]     (deg, setpoint)
            #   7-9    : vel[0..2]         (deg/s, LP-filtered từ ESP32)
            #  10-12   : vel_set[0..2]     (deg/s, feedforward)
            #  13-15   : acc[0..2]         (deg/s^2, SavGol từ vel LP — tính sẵn trong controller)
            #  16-18   : acc_set[0..2]     (deg/s^2, ref acceleration từ spline)
            #  19-21   : tau_out[0..2]     (Nm, torque gửi xuống ODrive, đã qua motor_sign)
            # Bỏ các field thừa (mot0/1/2_deg, raw joint, _last_tau_raw) để giảm overhead.
            # Khi cần debug sâu vẫn xem log trực tiếp từ _parse_line / _dynamic_calculation_locked.
            self.data.append((
                now,
                self.pos[0], self.pos[1], self.pos[2],
                self.pos_set[0], self.pos_set[1], self.pos_set[2],
                self.vel[0], self.vel[1], self.vel[2],
                float(self.vel_set[0]), float(self.vel_set[1]), float(self.vel_set[2]),
                float(self.acc[0]), float(self.acc[1]), float(self.acc[2]),
                float(self.acc_set[0]), float(self.acc_set[1]), float(self.acc_set[2]),   # ref acc (match ODESC)
                self._tau_out[0], self._tau_out[1], self._tau_out[2],
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
            # Auto-clear CAN errors khi phát hiện bus off
            if "CAN bus off" in line or "bus_off" in line.lower():
                print("[TWAI][AUTO] CAN bus off detected -> CLEAR 0/1/2")
                for _id in (0, 1, 2):
                    try:
                        if self.ser and self.ser.is_open:
                            self.ser.write(f"CLEAR {_id}\n".encode("ascii"))
                    except Exception as e:
                        print(f"[TWAI][AUTO] CLEAR {_id} failed: {e}")

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

        # Lưu ý: Block FB, thứ 2 đã được xóa (double-append bug).
        # Logic motion_armed đã được xử lý ở block FB đầu (line ~1016).

        elif line.startswith("INFO") or line.startswith("WARN: GOTO") or line.startswith("INFO: GOTO") or line.startswith("INFO: HOLD"):
            print(f"[TWAI] {line}")

        elif line.startswith("PRISM "):
            # ESP32 reply: PRISM hip_mm knee_mm l1_m l2_m
            try:
                parts = line.split()
                if len(parts) >= 5:
                    hip_mm = float(parts[1])
                    knee_mm = float(parts[2])
                    l1_m = float(parts[3])
                    l2_m = float(parts[4])
                    self.prismatic_hip_mm = hip_mm
                    self.prismatic_knee_mm = knee_mm
                    self.hip_link_length = l1_m
                    self.knee_link_length = l2_m
                    # Update CTC model
                    self.model3.prismatic_hip_mm = hip_mm
                    self.model3.prismatic_knee_mm = knee_mm
                    self.model3.update_prismatic_lengths()
                    if self.debug_serial_verbose:
                        print(f"[TWAI] PRISM: hip={hip_mm:.1f}mm l1={l1_m:.3f}m, knee={knee_mm:.1f}mm l2={l2_m:.3f}m")
            except (ValueError, IndexError):
                pass

        elif line.startswith("KOK,"):
            # Firmware báo calib limit switch EXT cho khớp knee (axis 1) hoàn tất.
            # Format: KOK,<q_max_deg>
            #   - q_max_deg = ROM khớp knee (firmware echo lại từ "HOME <q_max>" PC gửi xuống)
            # Firmware đã reset encoder count về 0 tại vị trí EXT → q_feedback = 0° khi duỗi.
            #   q_feedback = -count_deg          (count âm khi quay về phía FLEX)
            #   q_feedback = 0                    tại EXT (duỗi thẳng)
            #   q_feedback = -q_max               khi gập hết (phi_flex)
            try:
                parts = line.split(",")
                if len(parts) >= 2:
                    q_max_deg = float(parts[1])
                    axis = 1   # knee
                    with self.data_lock:
                        # EXT (count=0) = q_feedback=0° → joint_offset_deg[1] = 0
                        # Gập về phía flex → encoder count âm → joint_pos_deg[1] âm
                        self.joint_offset_deg[axis] = 0.0  # EXT làm gốc 0°
                        self.joint_pos_deg[axis] = 0.0
                        self.pos[axis] = 0.0
                        self.vel[axis] = 0.0
                        self.acc[axis] = 0.0
                        self._vel_lp_prev[axis] = 0.0
                        for buf in self._vel_med_buf:
                            buf.clear()
                        self._clear_vel_fit_buffers_locked()
                        self.isOffset = True
                        self.motion_armed = False
                        self._motion_hold = False
                        self._home_pending = False
                        self.q_max_deg = q_max_deg
                        self.status_message = f"Home knee OK — EXT=0°, q_max={q_max_deg:.1f}°"
                    print(f"[TWAI] KOK knee: q_max={q_max_deg:.3f}° → offset set (EXT=0°)")
            except (ValueError, IndexError) as e:
                print(f"[TWAI] KOK parse error: {line!r} → {e}")

        elif line.startswith("ERR:"):
            print(f"[TWAI][FIRMWARE] {line}")


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
        """Trả vị trí hiện tại (degrees) của cả 3 motor ở hệ world (pos + model_home_deg).

        setpoint GUI (gõ vào Entry khi Move) vẫn dùng hệ tương đối — dùng get_pos_relative()
        nếu cần giá trị thô cho setpoint/trajectory.
        """
        with self.data_lock:
            return (
                self.pos[0] + self.model_home_deg[0],
                self.pos[1] + self.model_home_deg[1],
                self.pos[2] + self.model_home_deg[2],
            )

    def get_pos_relative(self):
        """Trả vị trí hiện tại ở hệ tương đối (sau Set Offset) — dùng cho setpoint/trajectory."""
        with self.data_lock:
            return self.pos[0], self.pos[1], self.pos[2]

    def get_model_home(self):
        """Trả offset world frame mà CTC đang dùng (model_home_deg) — cho GUI hiển thị."""
        with self.data_lock:
            return (
                self.model_home_deg[0],
                self.model_home_deg[1],
                self.model_home_deg[2],
            )

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
            # Layout: (p0,p1,p2, kp0,kp1,kp2, kd0,kd1,kd2, ctrl_bw, enc_bw[, max_vel])
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
            if len(ctrlElms) >= 11:
                self.ctrl_bandwidth = float(ctrlElms[9])
                self.enc_bandwidth  = float(ctrlElms[10])
            if len(ctrlElms) >= 12:
                # Slot 11 = max trajectory velocity (deg/s) cho trajectory param_calc & on-board VMAX.
                self.max_vel = float(ctrlElms[11])

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
                # Áp dụng locked_axes: nếu axis bị khóa, giữ tại vị trí hiện tại (giống update_ctrlElms)
                for axis, locked in enumerate(self.locked_axes):
                    if locked:
                        if axis == 0:
                            p0_deg = self.pos[0]
                        elif axis == 1:
                            p1_deg = self.pos[1]
                        else:
                            p2_deg = self.pos[2]
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
            # Cập nhật CTC model — không quên ép static_friction mới vào model3
            # (CTC3Model là immutable field, phải gán lại tuple).
            if len(loadParms) >= 3 and hasattr(self, 'model3'):
                self.model3.static_friction = (self.static_friction, 0.0, self.static_friction)
                self.model3.coulomb_friction = (self.coul_friction, self.coul_friction, self.coul_friction)
            if len(loadParms) >= 4 and hasattr(self, 'model3'):
                self.model3.viscous_friction = (self.visc_friction, self.visc_friction, self.visc_friction)
            self._recalc_plant_mass_inertia()

    def clear_error(self):
        self.error = False
        # Gửi lệnh CLEAR xuống ESP32 bridge để xóa errors ở cả 3 ODrive.
        # Bridge sẽ forward tới từng ODrive qua CAN (ClearErrors command).
        if self.esp32_ready:
            self._send_simple_cmd("CLEAR")
        # Trả về trạng thái hiện tại (False ngay lập tức — errors thật sẽ được
        # phát hiện lại qua FB heartbeat/error register).
        return self.error

    def _clear_odrive_errors(self, why: str = ""):
        """Clear ODrive errors qua ESP32 bridge. Dùng trước các thao tác 'state-changing'
        (Enable, Set Offset, Reset) để đảm bảo ODrive không còn ERROR state cũ.

        An toàn khi gọi nhiều lần — idempotent.
        """
        if not self.esp32_ready:
            # ESP32 chưa ready: không thể gửi CLEAR, nhưng vẫn reset flag PC.
            # GUI sẽ retry khi ESP32 ready.
            self.error = False
            return
        try:
            self._send_simple_cmd("CLEAR")
            if why:
                print(f"[TWAI] CLEAR errors sent ({why})")
        except Exception as e:
            print(f"[TWAI] CLEAR error failed ({why}): {e}")
        # Reset flag PC-side ngay (không đợi ESP32 phản hồi).
        self.error = False

    def set_torque_mode(self, enabled: bool = True):
        self.use_torque_mode = enabled
        if self.esp32_ready:
            self._send_simple_cmd("TORQUE" if enabled else "POSITION")

    def go_home(self, q_max_deg: float | None = None):
        """Yêu cầu firmware calib homing cho khớp knee (axis 1) bằng limit switch EXT.

        Firmware quay motor dir=+1 (cùng chiều moment dương, toward duỗi thẳng) cho đến khi
        chạm GPIO 41 (công tắc duỗi), brake, reset encoder count = 0 tại đó, gửi
        "KOK,<phi_ext>,<q_max>" lên PC.

        Args:
            q_max_deg: ROM khớp knee (deg). Nếu None, dùng self.q_max_default (90°).
                       Sau calib, q_feedback (knee) = phi_ext - encoder_raw (deg).
                       q_feedback = -q_max khi gập hết.
        """
        if not self.esp32_ready:
            self.status_message = "ESP32 chưa sẵn sàng"
            return
        if q_max_deg is None:
            q_max_deg = self.q_max_default
        with self.data_lock:
            self.motion_armed = False
            self._motion_hold = False
            self._home_pending = True
            self._setpoint_dirty = False
            self._was_motion_armed = False
            self._motion_time_active = False
            self._tau_out[0] = self._tau_out[1] = self._tau_out[2] = 0.0
            self.isOffset = False   # reset cho đến khi KOK
            self.q_max_deg = q_max_deg
        # Gửi kèm q_max_deg để firmware hiển thị + gửi lại cho PC qua KOK
        self._send_simple_cmd(f"HOME {q_max_deg:.1f}")
        self.status_message = f"Homing knee (q_max={q_max_deg:.1f}°) — seek EXT(GPIO41)"

    def cancel_home(self):
        """Hủy homing đang chạy (gửi HOMECANCEL cho firmware)."""
        if not self.esp32_ready:
            return
        self._send_simple_cmd("HOMECANCEL")
        with self.data_lock:
            self._home_pending = False
            self.motion_armed = False
        self.status_message = "Homing cancelled"

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
                    # Chờ thông minh — nếu là PermissionError (đang chiếm port) thì chờ lâu hơn
                    wait_s = self._connect_retry_delay_s if self.error else 0.5
                    self._stop_event.wait(wait_s)
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
                # DEBUG: log state mỗi 1s để xác minh tại sao torque không gửi xuống ESP32.
                if not hasattr(self, "_last_state_log_ts"):
                    self._last_state_log_ts = 0.0
                if now - self._last_state_log_ts >= 1.0:
                    self._last_state_log_ts = now
                    print(
                        f"[TWAI][STATE] ctrl={self.is_controlable()} "
                        f"armed={self.motion_armed} was_armed={self._was_motion_armed} "
                        f"torque_mode={self.use_torque_mode} on_board={self.on_board_ctl} "
                        f"locked={self.locked_axes}"
                    )
                if self.is_controlable() and self.motion_armed:
                    now_wall = time.time()
                    if self.friction_only_mode:
                        axis = self.friction_only_axis if self.friction_only_axis is not None else 0
                        tau = self.friction_only_torque
                        self._send_torque(0, 0.0)
                        self._send_torque(1, 0.0)
                        self._send_torque(2, 0.0)
                        tau_motor = self._joint_to_motor_torque(axis, tau)
                        tau_motor *= self.motor_sign[axis]
                        self._send_torque(axis, tau_motor)
                        self.torque_set[0] = 0.0
                        self.torque_set[1] = 0.0
                        self.torque_set[2] = 0.0
                        self.torque_set[axis] = tau_motor
                        if now_wall - self._last_torque_log_ts >= 1.0:
                            print(f"[TWAI] Friction-only mode axis={axis} tau={tau:.6f}")
                            self._last_torque_log_ts = now_wall
                        continue

                    # ODESC-style: restart spline khi user nhấn Run (mới hoặc lại).
                    #   arm_edge = rising edge của motion_armed (lần đầu)
                    #   new_motion_pending = GUI đã gọi _start_new_motion() (lần nhấn Run kế tiếp,
                    #     trong khi motion_armed vẫn True sau move trước)
                    arm_edge = self.motion_armed and not self._was_motion_armed
                    restart_edge = arm_edge or self._new_motion_pending
                    if restart_edge:
                        self._motion_clock_start()
                        self._new_motion_pending = False
                        # === ON-BOARD MODE: gửi GOTO thay vì tính torque trên PC ===
                        with self.data_lock:
                            targets = (self._last_motion_targets[0],
                                       self._last_motion_targets[1],
                                       self._last_motion_targets[2])
                            kp = self.Kp_axes[:]; kd = self.Kd_axes[:]
                            vmax = self.max_vel
                        # Sync gain & max_vel với ESP32 (rất nhẹ, ~30 bytes)
                        self.set_gains(kp[0], kp[1], kp[2], kd[0], kd[1], kd[2])
                        self.set_max_vel(vmax)
                        self.goto(targets[0], targets[1], targets[2])
                        print(f"[TWAI][ON-BOARD] GOTO targets={targets} vmax={vmax}")
                        # Đánh dấu đã gửi — ESP32 sẽ tự báo motion_active qua dòng 'S,...'
                        self._was_motion_armed = True
                        continue

                    # tau từ ESP32 feedback đã được parse ở trên - giữ nguyên
                    # Khi on_board_ctl=True, ESP32 tính torque
                    # Khi on_board_ctl=False, tau đã = 0.0 từ line 928
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
                            # ON-BOARD: gửi HOLD để ESP32 dừng
                            if self.on_board_ctl:
                                self._send_simple_cmd("HOLD")
                        elif self.use_torque_mode:
                            if self.on_board_ctl:
                                # ESP32 đã tính torque, PC chỉ giám sát
                                tau0 = tau1 = tau2 = 0.0
                                if trace_now:
                                    print("[TWAI][ON-BOARD] ESP32 đang điều khiển torque")
                            else:
                                # PC tính torque (chế độ cũ)
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
                        # ── ODESC-style: KHÔNG disarm ở PC mode ──
                        # CTC vẫn chạy mỗi tick với _motion_time_active=False → startup_t=0
                        # → w=0 → G_smooth = G_hold (full gravity comp) → giữ pos tự nhiên.
                        # Khi user nhấn Run với target mới, GUI gọi _start_new_motion() để
                        # restart _motion_t0 (giống ODESC update_ctrlElms → t_ref = time.time()).
                        self._motion_time_active = False
                        if self.on_board_ctl:
                            # On-board: ESP32 đã chạy xong trajectory, tắt motion_active ở firmware.
                            self._reset_torque_slew()
                            self._send_torque(0, 0.0)
                            self._send_torque(1, 0.0)
                            self._send_torque(2, 0.0)
                            self.motion_armed = False
                            self._was_motion_armed = False
                            self._motion_hold = True
                            self._setpoint_dirty = False
                            # Gửi HOLD để ESP32 tắt motion_active (tránh firmware "đông cứng"
                            # ở motion=1 và gây re-arm GOTO vô tận).
                            self._send_simple_cmd("HOLD")
                        else:
                            # PC mode: giữ CTC chạy tiếp với G_hold blend để giữ pos.
                            # Loop tick tiếp theo sẽ vào nhánh motion_armed=True (vì không disarm)
                            # → _refresh_traj_refs_locked() với t_prog=0 → pos_set = end_state,
                            #   _dynamic_calculation_locked() → tau = G_hold (gravity comp).
                            self._motion_hold = True
                        self.status_message = "Motion completed (holding CTC)"
                    elif self.use_torque_mode and not self.on_board_ctl:
                        # Chỉ gửi torque khi PC điều khiển (không phải on_board_ctl)
                        tau0_motor = self._joint_to_motor_torque(0, tau0)
                        tau1_motor = self._joint_to_motor_torque(1, tau1)
                        tau2_motor = self._joint_to_motor_torque(2, tau2)
                        # Apply motor_sign để bù lắp ngược chiều (motor 1 = knee).
                        # Lý do: motor 1 lắp ngược với sign convention của CTC model, nên
                        # torque joint-side dương → motor phải quay chiều âm để đúng ngữ nghĩa.
                        # Plot cũng bỏ đảo tương ứng ở data.append (chỉ hiển thị giá trị gửi xuống ODrive).
                        tau0_motor *= self.motor_sign[0]
                        tau1_motor *= self.motor_sign[1]
                        tau2_motor *= self.motor_sign[2]
                        tau0_sent = self._slew_limited_torque(0, tau0_motor)
                        tau1_sent = self._slew_limited_torque(1, tau1_motor)
                        tau2_sent = self._slew_limited_torque(2, tau2_motor)
                        self._tau_sent = (tau0_sent, tau1_sent, tau2_sent)
                        self._send_torque(0, tau0_sent)
                        self._send_torque(1, tau1_sent)
                        self._send_torque(2, tau2_sent)
                        # Cảnh báo nếu CTC yêu cầu torque vượt max_torque → bão hoà → motor lag.
                        if now_wall - self._last_torque_log_ts >= 1.0:
                            for axis, (pre, sent) in enumerate(zip(
                                (tau0_motor, tau1_motor, tau2_motor),
                                (tau0_sent, tau1_sent, tau2_sent),
                            )):
                                if abs(pre - sent) > 1e-3:
                                    print(
                                        f"[TWAI] ⚠ axis {axis} CLIP motor tau "
                                        f"pre={pre:+.3f} → sent={sent:+.3f} Nm "
                                        f"(max_torque=±{self.max_torque:.2f} Nm)"
                                    )
                            print(
                                "[TWAI] CTC torque cmd "
                                f"tau_raw=({self._last_tau_raw[0]:.4f}, {self._last_tau_raw[1]:.4f}, {self._last_tau_raw[2]:.4f}) "
                                f"tau_joint_cmd=({tau0:.4f}, {tau1:.4f}, {tau2:.4f}) "
                                f"tau_motor_cmd=({tau0_sent:.4f}, {tau1_sent:.4f}, {tau2_sent:.4f}) "
                            )
                            self._last_torque_log_ts = now_wall
                    else:
                        if not self.on_board_ctl:
                            # Chỉ gửi position khi PC điều khiển trực tiếp
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
                    if not self.on_board_ctl:
                        self._send_torque(0, 0.0)
                        self._send_torque(1, 0.0)
                        self._send_torque(2, 0.0)
                self._was_motion_armed = self.motion_armed

            except Exception as e:
                import traceback
                print(f"[TWAI] Lỗi vòng lặp: {e}\n{traceback.format_exc()}")
                self.connected = False
                self.error     = True
                self._stop_event.wait(1.0)

            # ── Giữ ~100Hz ────────────────────────────────────────────────
            elapsed = time.perf_counter() - t_start
            sleep_t = 0.01 - elapsed
            if sleep_t > 0:
                self._stop_event.wait(sleep_t)

        print("[TWAI] Thread dừng.")
