import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import logging
import math
from collections import deque
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import twai_serial_controller1 as twai_controller

try:
    import trajectory_controller as odrive_backend
except Exception:
    odrive_backend = None

try:
    from pso_tuner import optimize_kp_kd
except Exception:
    optimize_kp_kd = None

try:
    from kinematic import get_acc_jerk
except Exception:
    get_acc_jerk = None

try:
    from system_identifier import estimate_first_order_model
except Exception:
    estimate_first_order_model = None

# ── State aliases ────────────────────────────────────────────────────────────
IDLE                  = twai_controller.IDLE
CLOSED_LOOP_CONTROL   = twai_controller.CLOSED_LOOP_CONTROL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CTCControlGUI")

UPDATE_INTERVAL_MS = 33   # ~30 Hz GUI update — đủ mượt với mắt, giảm tải so với 50ms nhưng
                         # vẫn mượt hơn 20Hz; ODESC dùng 10ms nhưng plot đơn giản hơn nhiều.


# ════════════════════════════════════════════════════════════════════════════
# Connection Dialog
# ════════════════════════════════════════════════════════════════════════════

class ConnectDialog(tk.Toplevel):

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Kết nối Controller")
        self.resizable(False, False)
        self.grab_set()

        self.result_port = None
        self.result_baudrate = None
        self.result_backend = None

        # ── Widgets ──────────────────────────────────────────────────────
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Backend:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.backend_var = tk.StringVar(value=parent.backend_var.get())
        backend_combo = ttk.Combobox(frame, textvariable=self.backend_var,
                                     values=["ODrive (Torque/CTC)", "TWAI"], width=18, state="readonly")
        backend_combo.grid(row=0, column=1, padx=8, pady=4)

        ttk.Label(frame, text="COM Port:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ports = twai_controller.list_serial_ports()
        self.port_var = tk.StringVar(value=ports[0] if ports else "COM3")
        port_combo = ttk.Combobox(frame, textvariable=self.port_var,
                                  values=ports, width=14)
        port_combo.grid(row=1, column=1, padx=8, pady=4)

        ttk.Label(frame, text="Baudrate:").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.baud_var = tk.StringVar(value="115200")
        baud_combo = ttk.Combobox(frame, textvariable=self.baud_var,
                                  values=["115200", "921600", "230400"], width=14)
        baud_combo.grid(row=2, column=1, padx=8, pady=4)



        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(btn_frame, text="Kết nối", command=self._ok).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Hủy", command=self.destroy).pack(side=tk.LEFT, padx=6)

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window()

    def _ok(self):
        self.result_backend = self.backend_var.get().strip()
        self.result_port = self.port_var.get().strip()
        self.result_baudrate = int(self.baud_var.get().strip())
        self.destroy()


# ════════════════════════════════════════════════════════════════════════════
# Main GUI
# ════════════════════════════════════════════════════════════════════════════

class ControlGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Motor Controller — CTC Torque Control")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Controller (sẽ được khởi tạo sau khi chọn backend) ──────────
        self.ctrl = None
        self.backend_var = tk.StringVar(value="ODrive (Torque/CTC)")
        self.backend_var = tk.StringVar(value="ODrive (Torque/CTC)")

        # ── Params mặc định cho control panel ───────────────────────────
        self._default_ctrl = {
            "Target Motor 0": (0.0,   "deg"),
            "Target Motor 1": (0.0,   "deg"),
            "Target Motor 2": (0.0,   "deg"),
            "Kp0":            (3.0,   None),
            "Kp1":            (15.0,  None),
            "Kp2":            (3.0,   None),
            "Kd0":            (1.0,   None),
            "Kd1":            (3.0,   None),
            "Kd2":            (1.0,   None),
            "Control bandwidth": (1200.0, None),
            "Encoder bandwidth": (100.0,   None),
            "Max Trajectory Velocity": (60.0, "deg/s"),
        }
        self._ctrl_index = {}
        self.trajectory_var = tk.StringVar(value="Spline")
        self._last_running_targets: tuple[float, float, float] | None = None
        self._default_load = {
            "External load":   (0.0,   "kg"),
            "Load position":   (0.0,   "m"),
            "Coulomb friction":(0.0,   "Nm"),
            "Viscous friction":(0.0,  "Nm/(rad/s)"), #0.00276
        }
        # Cho phép điều chỉnh dải nhập load ngay tại GUI.
        # Mỗi tham số: (min, max, step).
        # Range rộng để cover cả kg load thật lẫn giá trị friction đặc trưng.
        # (Bản cũ set (0,0,0) khiến mọi input ≠ 0 bị reject.)
        self._load_input_cfg = {
            "External load":    (0.0, 0.0, 0.0),
            "Load position":    (0.0, 0.0, 0.0),
            "Coulomb friction": (0.0, 0.0, 0.0),
            "Viscous friction": (0.0, 0.0, 0.0),
        }
        # ── Prismatic joint parameters (từ VL53L0X) ─────────────────────────
        self._default_prismatic = {
            "Hip prismatic (mm)": (350.0, "mm"),
            "Knee prismatic (mm)": (350.0, "mm"),
        }
        self._prismatic_cfg = {
            "Hip prismatic (mm)": (350.0, 450.0, 1.0),
            "Knee prismatic (mm)": (350.0, 450.0, 1.0),
        }

        self.plotting  = True
        self._last_t0  = None

        self._build_ui()
        self.after(UPDATE_INTERVAL_MS, self._update)

        # ── Mở connection dialog ngay khi khởi động 
        self.after(100, self._open_connect_dialog)

    def _build_ui(self):
        main = ttk.Frame(self, padding=6)
        main.pack(fill=tk.BOTH, expand=True)

        left  = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Right panel: scrollable ─────────────────────────────────────────
        self._right_canvas = tk.Canvas(main, width=380, bg="#F0F0F0", highlightthickness=0)
        self._right_scroll = ttk.Scrollbar(main, orient=tk.VERTICAL, command=self._right_canvas.yview)
        self._right_canvas.configure(yscrollcommand=self._right_scroll.set)

        self._right_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._right_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Frame bên trong canvas để chứa nội dung
        self._right_frame = ttk.Frame(self._right_canvas)
        self._right_canvas.create_window((0, 0), window=self._right_frame, anchor="nw")

        # Cập nhật scrollregion khi frame con thay đổi kích thước
        def _on_frame_configure(event):
            self._right_canvas.configure(scrollregion=self._right_canvas.bbox("all"))

        self._right_frame.bind("<Configure>", _on_frame_configure)

        # Cho phép cuộn bằng chuột giữa/mousewheel
        def _on_mousewheel(event):
            self._right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self._right_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Plot ─────────────────────────────────────────────────────────
        self.fig = Figure(figsize=(8, 12), dpi=100)
        gs = self.fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
        self.ax_pos = self.fig.add_subplot(gs[0])
        self.ax_vel = self.fig.add_subplot(gs[1])
        self.ax_acc = self.fig.add_subplot(gs[2])
        self.ax_tau = self.fig.add_subplot(gs[3])
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Lines
        self._line_pos0,     = self.ax_pos.plot([], [], label="Motor 0 (deg, world)", color="#2196F3")
        self._line_pos1,     = self.ax_pos.plot([], [], label="Motor 1 (deg, world)", color="#FF9800")
        self._line_pos2,     = self.ax_pos.plot([], [], label="Motor 2 (deg, world)", color="#4CAF50")
        # Setpoint cũng ở world để cùng hệ với pos — so sánh và err đúng.
        self._line_pos0_set, = self.ax_pos.plot([], [], label="M0 Setpoint",   color="#2196F3",
                                                linestyle="--", alpha=0.6)
        self._line_pos1_set, = self.ax_pos.plot([], [], label="M1 Setpoint",   color="#FF9800",
                                                linestyle="--", alpha=0.6)
        self._line_pos2_set, = self.ax_pos.plot([], [], label="M2 Setpoint",   color="#4CAF50",
                                                linestyle="--", alpha=0.6)
        # Velocity lines in second subplot (đã thay Error)
        self._line_vel0, = self.ax_vel.plot([], [], label="M0 actual", color="#2196F3")
        self._line_vel0_set, = self.ax_vel.plot([], [], label="M0 setpoint", color="#2196F3", linestyle="--", alpha=0.7)
        self._line_vel1, = self.ax_vel.plot([], [], label="M1 actual", color="#FF9800")
        self._line_vel1_set, = self.ax_vel.plot([], [], label="M1 setpoint", color="#FF9800", linestyle="--", alpha=0.7)
        self._line_vel2, = self.ax_vel.plot([], [], label="M2 actual", color="#4CAF50")
        self._line_vel2_set, = self.ax_vel.plot([], [], label="M2 setpoint", color="#4CAF50", linestyle="--", alpha=0.7)

        # Acceleration lines in third subplot (actual from get_acc_jerk of vel, ref from get_acc_jerk of vel_set)
        self._line_acc0, = self.ax_acc.plot([], [], label="Accel M0 actual", color="#2196F3")
        self._line_acc1, = self.ax_acc.plot([], [], label="Accel M1 actual", color="#FF9800")
        self._line_acc2, = self.ax_acc.plot([], [], label="Accel M2 actual", color="#4CAF50")
        self._line_acc0_set, = self.ax_acc.plot([], [], label="Accel M0 ref", color="#2196F3", linestyle="--", alpha=0.6)
        self._line_acc1_set, = self.ax_acc.plot([], [], label="Accel M1 ref", color="#FF9800", linestyle="--", alpha=0.6)
        self._line_acc2_set, = self.ax_acc.plot([], [], label="Accel M2 ref", color="#4CAF50", linestyle="--", alpha=0.6)

        self._line_tau0, = self.ax_tau.plot([], [], label="M0 actual τ (Nm)", color="#2196F3", linewidth=1.5)
        self._line_tau1, = self.ax_tau.plot([], [], label="M1 actual τ (Nm)", color="#FF9800", linewidth=1.5)
        self._line_tau2, = self.ax_tau.plot([], [], label="M2 actual τ (Nm)", color="#4CAF50", linewidth=1.5)

        self._line_tauctc0, = self.ax_tau.plot([], [], label="M0 CTC τ (Nm)", color="#2196F3", linestyle="--", alpha=0.6)
        self._line_tauctc1, = self.ax_tau.plot([], [], label="M1 CTC τ (Nm)", color="#FF9800", linestyle="--", alpha=0.6)
        self._line_tauctc2, = self.ax_tau.plot([], [], label="M2 CTC τ (Nm)", color="#4CAF50", linestyle="--", alpha=0.6)

        self.ax_pos.set_ylabel("Position (deg)")
        self.ax_pos.grid(True);  self.ax_pos.legend(loc="upper right", fontsize=7)
        self.ax_vel.set_ylabel("Velocity (deg/s)")
        self.ax_vel.grid(True);  self.ax_vel.legend(loc="upper right", fontsize=7)
        self.ax_acc.set_ylabel("Accel (deg/s²)")
        self.ax_acc.grid(True);  self.ax_acc.legend(loc="upper right", fontsize=7)
        self.ax_tau.set_ylabel("Torque (Nm)")
        self.ax_tau.set_xlabel("Time (s)")
        self.ax_tau.grid(True);  self.ax_tau.legend(loc="upper right", fontsize=7)

        # ── Action buttons ────────────────────────────────────────────────
        top_right = ttk.Frame(self._right_frame, padding=6)
        top_right.pack(side=tk.TOP, fill=tk.X)
        for c in range(3): top_right.columnconfigure(c, weight=1)

        # Status indicator
        self.status_label = tk.Label(top_right, text="Chưa kết nối",
                                     relief="ridge", bg="lightgrey", wraplength=110)
        self.status_label.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)

        # Connect button
        self.btn_connect = tk.Button(top_right, text="Kết nối...",
                                     bg="#4CAF50", fg="white", relief="raised",
                                     command=self._open_connect_dialog)
        self.btn_connect.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)

        self.btn_optimize_pso = ttk.Button(top_right, text="PSO Kp/Kd", command=self._on_optimize_pso, state="disabled")
        self.btn_optimize_pso.grid(row=0, column=3, sticky="nsew", padx=2, pady=2)

        # Offset
        self.btn_offset = tk.Button(top_right, text="Set Offset",
                                    bg="tomato", relief="raised", command=self._on_offset)
        self.btn_offset.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # Close Loop toggle
        self.btn_mode = tk.Button(top_right, text="Close Loop",
                                  bg="lightgreen", relief="raised", command=self._on_mode_tog)
        self.btn_mode.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)

        # Stop/Continue plotting
        self.btn_plot = tk.Button(top_right, text="Stop Plot",
                                  relief="raised", command=self._on_toggle_plot)
        self.btn_plot.grid(row=1, column=2, sticky="nsew", padx=2, pady=2)

        # Reset
        self.btn_reset = tk.Button(top_right, text="Reset",
                                   relief="raised", command=self._on_reset)
        self.btn_reset.grid(row=2, column=0, sticky="nsew", padx=2, pady=2)

        # Home
        self.btn_home = tk.Button(top_right, text="Home",
                                  bg="dodgerblue", fg="white", relief="raised",
                                  command=self._on_home)
        self.btn_home.grid(row=2, column=1, sticky="nsew", padx=2, pady=2)

        # ESTOP
        self.btn_estop = tk.Button(top_right, text="ESTOP",
                                   bg="red", fg="white", relief="raised",
                                   command=self._on_estop)
        self.btn_estop.grid(row=2, column=2, sticky="nsew", padx=2, pady=2)

        # ── Control Panel  ────────────────────────────────────────────────
        ctrl_frame = ttk.LabelFrame(self._right_frame, text="Điều Khiển", padding=8)
        ctrl_frame.pack(padx=6, pady=6, fill=tk.X)

        ctrl_grid = ttk.Frame(ctrl_frame)
        ctrl_grid.pack(fill=tk.X)

        # Current positions (read-only)
        ttk.Label(ctrl_grid, text="Pos Motor 0 (deg):").grid(row=0, column=0, sticky=tk.W)
        self.entry_pos0 = ttk.Entry(ctrl_grid, width=12, state="readonly")
        self.entry_pos0.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(ctrl_grid, text="Pos Motor 1 (deg):").grid(row=1, column=0, sticky=tk.W)
        self.entry_pos1 = ttk.Entry(ctrl_grid, width=12, state="readonly")
        self.entry_pos1.grid(row=1, column=1, padx=4, pady=2)

        ttk.Label(ctrl_grid, text="Pos Motor 2 (deg):").grid(row=2, column=0, sticky=tk.W)
        self.entry_pos2 = ttk.Entry(ctrl_grid, width=12, state="readonly")
        self.entry_pos2.grid(row=2, column=1, padx=4, pady=2)

        # Editable control params
        self.control_panel = []
        for i, (key, (val, unit)) in enumerate(self._default_ctrl.items()):
            row_idx = i + 3
            label_text = f"{key} ({unit}):" if unit else f"{key}:"
            ttk.Label(ctrl_grid, text=label_text).grid(row=row_idx, column=0, sticky=tk.W, pady=1)
            v = tk.StringVar(value=f"{val:.2f}")
            entry = ttk.Entry(ctrl_grid, textvariable=v, width=12)
            entry.grid(row=row_idx, column=1, padx=4, pady=2)
            self._ctrl_index[key] = len(self.control_panel)
            self.control_panel.append([entry, v])

        # Trajectory mode selector
        traj_frame = ttk.LabelFrame(ctrl_frame, text="Trajectory", padding=4)
        traj_frame.pack(pady=(4, 0), fill=tk.X)
        ttk.Label(traj_frame, text="Mode:").pack(side=tk.LEFT, padx=(0, 6))
        self.trajectory_combo = ttk.Combobox(
            traj_frame,
            textvariable=self.trajectory_var,
            values=["Spline", "Quintic", "Cubic", "Trapezoidal"],
            state="readonly",
            width=12,
        )
        self.trajectory_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.trajectory_combo.bind("<<ComboboxSelected>>", self._on_trajectory_mode_changed)
        self.btn_apply_traj = ttk.Button(traj_frame, text="Apply Trajectory", command=self._apply_trajectory_mode)
        self.btn_apply_traj.pack(side=tk.LEFT, padx=(6, 0))

        # Auto-calc Kp/Kd from bandwidth
        self.btn_apply_bw = ttk.Button(ctrl_frame, text="Auto Calc Kp/Kd",
                                       command=self._apply_bandwidth)
        self.btn_apply_bw.pack(pady=(4, 0), fill=tk.X)

        # Optimize Kp/Kd by PSO
        self.btn_optimize_pso_ctrl = ttk.Button(ctrl_frame, text="Optimize Kp/Kd by PSO",
                                                command=self._on_optimize_pso)
        self.btn_optimize_pso_ctrl.pack(pady=(4, 0), fill=tk.X)

        # Axis lock toggles
        self.axis_lock_vars = [tk.BooleanVar(value=False) for _ in range(3)]
        lock_frame = ttk.LabelFrame(ctrl_frame, text="Axis Lock", padding=4)
        lock_frame.pack(pady=(4, 0), fill=tk.X)
        for axis in range(3):
            cb = ttk.Checkbutton(
                lock_frame,
                text=f"Lock Motor {axis}",
                variable=self.axis_lock_vars[axis],
                command=self._apply_axis_locks,
            )
            cb.pack(anchor=tk.W)

        # Move button
        self.btn_move = ttk.Button(ctrl_frame, text="▶  Run CTC Motion",
                                   command=self._on_move, state="disabled")
        self.btn_move.pack(pady=(6, 0), fill=tk.X)

        # ── Load Parameters ───────────────────────────────────────────────
        param_frame = ttk.LabelFrame(self._right_frame, text="Parameters", padding=8)
        param_frame.pack(padx=6, pady=6, fill=tk.X)

        param_grid = ttk.Frame(param_frame)
        param_grid.pack(fill=tk.X)

        self.param_panel = []
        for i, (key, (val, unit)) in enumerate(self._default_load.items()):
            ttk.Label(param_grid,
                      text=f"{key} ({unit}):" if unit else f"{key}:").grid(
                row=i, column=0, sticky=tk.W, pady=1)
            v = tk.StringVar(value=f"{val:.3f}")
            min_v, max_v, step_v = self._load_input_cfg.get(key, (-1e9, 1e9, 0.01))
            entry = tk.Spinbox(
                param_grid,
                textvariable=v,
                from_=min_v,
                to=max_v,
                increment=step_v,
                width=10,
                format="%.3f",
            )
            entry.grid(row=i, column=1, padx=4, pady=2)
            self.param_panel.append([entry, v, key])

        self.btn_send_param = ttk.Button(param_frame, text="Gửi Parameters",
                                         command=self._on_send_params, state="disabled")
        self.btn_send_param.pack(pady=(6, 0), fill=tk.X)

        # ── Prismatic Joint Parameters (VL53L0X) ─────────────────────────
        prism_frame = ttk.LabelFrame(self._right_frame, text="Prismatic Joints (mm)", padding=8)
        prism_frame.pack(padx=6, pady=6, fill=tk.X)

        self._prism_vars = {}
        for i, (key, (val, unit)) in enumerate(self._default_prismatic.items()):
            ttk.Label(prism_frame, text=f"{key}:").grid(
                row=i, column=0, sticky=tk.W, pady=1)
            min_v, max_v, step_v = self._prismatic_cfg.get(key, (350.0, 450.0, 1.0))
            v = tk.StringVar(value=f"{val:.1f}")
            entry = tk.Spinbox(
                prism_frame,
                textvariable=v,
                from_=min_v,
                to=max_v,
                increment=step_v,
                width=10,
                format="%.1f",
            )
            entry.grid(row=i, column=1, padx=4, pady=2)
            self._prism_vars[key] = v

        self.btn_send_prismatic = ttk.Button(prism_frame, text="Gửi Prismatic",
                                              command=self._on_send_prismatic, state="disabled")
        self.btn_send_prismatic.grid(row=len(self._default_prismatic), column=0, columnspan=2, pady=(6, 0), sticky="ew")

        # ── COM Translation Parameters (delta_x) ──────────────────────────
        delta_frame = ttk.LabelFrame(self._right_frame, text="COM Translation (m)", padding=8)
        delta_frame.pack(padx=6, pady=6, fill=tk.X)

        ttk.Label(delta_frame, text="Hip delta_x (m):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self._delta_x1_var = tk.StringVar(value="0.000")
        ttk.Entry(delta_frame, textvariable=self._delta_x1_var, width=12).grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(delta_frame, text="Knee delta_x (m):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self._delta_x2_var = tk.StringVar(value="0.000")
        ttk.Entry(delta_frame, textvariable=self._delta_x2_var, width=12).grid(row=1, column=1, padx=4, pady=2)

        self.btn_send_delta = ttk.Button(delta_frame, text="Cập nhật COM",
                                         command=self._on_send_delta_x, state="disabled")
        self.btn_send_delta.grid(row=2, column=0, columnspan=2, pady=(6, 0), sticky="ew")

        # ── Cylinder Control (Manual) ─────────────────────────────────────
        cyl_frame = ttk.LabelFrame(self._right_frame, text="Điều khiển Cylinder (Laser)", padding=8)
        cyl_frame.pack(padx=6, pady=6, fill=tk.X)

        # Laser readings display
        ttk.Label(cyl_frame, text="Hip laser (mm):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self._laser_hip_var = tk.StringVar(value="--")
        ttk.Label(cyl_frame, textvariable=self._laser_hip_var, font=("Arial", 10, "bold")).grid(row=0, column=1, sticky=tk.W, padx=4)

        ttk.Label(cyl_frame, text="Knee laser (mm):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self._laser_knee_var = tk.StringVar(value="--")
        ttk.Label(cyl_frame, textvariable=self._laser_knee_var, font=("Arial", 10, "bold")).grid(row=1, column=1, sticky=tk.W, padx=4)

        # Hip cylinder buttons
        ttk.Label(cyl_frame, text="Hip:").grid(row=2, column=0, sticky=tk.W, pady=(8, 2))
        btn_hip_plus = ttk.Button(cyl_frame, text="↑ Kéo dài",
                                  command=lambda: self._on_cylinder(0, 1), state="disabled")
        btn_hip_plus.grid(row=2, column=1, padx=2, pady=(8, 2), sticky="ew")
        btn_hip_minus = ttk.Button(cyl_frame, text="↓ Thu ngắn",
                                   command=lambda: self._on_cylinder(0, -1), state="disabled")
        btn_hip_minus.grid(row=3, column=1, padx=2, pady=2, sticky="ew")
        self._btn_hip_plus = btn_hip_plus
        self._btn_hip_minus = btn_hip_minus

        # Knee cylinder buttons
        ttk.Label(cyl_frame, text="Knee:").grid(row=4, column=0, sticky=tk.W, pady=(8, 2))
        btn_knee_plus = ttk.Button(cyl_frame, text="↑ Kéo dài",
                                    command=lambda: self._on_cylinder(1, 1), state="disabled")
        btn_knee_plus.grid(row=4, column=1, padx=2, pady=(8, 2), sticky="ew")
        btn_knee_minus = ttk.Button(cyl_frame, text="↓ Thu ngắn",
                                    command=lambda: self._on_cylinder(1, -1), state="disabled")
        btn_knee_minus.grid(row=5, column=1, padx=2, pady=2, sticky="ew")
        self._btn_knee_plus = btn_knee_plus
        self._btn_knee_minus = btn_knee_minus

        # ── Status bar ────────────────────────────────────────────────────
        status_frame = ttk.Frame(self._right_frame, padding=6)
        status_frame.pack(fill=tk.X)

        self.status_text = tk.StringVar(value="Status: chưa kết nối")
        ttk.Label(status_frame, textvariable=self.status_text,
                  relief=tk.RIDGE).pack(fill=tk.X)

        self.status_text = tk.StringVar(value="Status: chưa kết nối")

    def _open_connect_dialog(self):
        """Mở dialog chọn backend/COM, tạo controller mới và start thread."""
        if self.ctrl is not None:
            try:
                self.ctrl.stop()
                self.ctrl.join(timeout=2.0)
            except Exception:
                pass
            self.ctrl = None

        dlg = ConnectDialog(self)
        if dlg.result_backend is None:
            return

        self.backend_var.set(dlg.result_backend)

        if dlg.result_backend == "ODrive (Torque/CTC)" and odrive_backend is not None:
            self.ctrl = odrive_backend.ODriveThread()
        elif dlg.result_backend == "TWAI":
            self.ctrl = twai_controller.TWAIController(
                serial_port=dlg.result_port,
                baudrate=dlg.result_baudrate,
                debug_serial_verbose=False,
            )
        else:
            messagebox.showerror("Lỗi", "Backend ODrive chưa sẵn sàng hoặc không import được.")
            return

        self.ctrl.start()
        self.status_text.set(f"Status: đang kết nối {dlg.result_backend}...")
        logger.info(f"Controller started: backend={dlg.result_backend}")
        self.btn_optimize_pso.configure(state="normal")
        self.btn_optimize_pso_ctrl.configure(state="normal")

    # ════════════════════════════════════════════════════════════════════════
    # Button callbacks
    # ════════════════════════════════════════════════════════════════════════

    def _on_offset(self):
        if self.ctrl:
            try:
                self.ctrl.set_offset()
                # Reset trục thời gian plot để vẽ từ thời điểm Set Offset (entry = model_home_deg).
                self._last_t0 = None
                self.btn_offset.configure(state="disabled", bg="lightgreen")
                self.status_text.set("Status: offset đã set")
            except Exception:
                logger.exception("Offset error")

    def _on_toggle_plot(self):
        self.plotting = not self.plotting
        self.btn_plot.config(text="Stop Plot" if self.plotting else "Continue Plot")

    def _on_estop(self):
        if self.ctrl:
            try:
                self.ctrl.emergency_stop()
                self.btn_estop.config(state="disabled")
                self.status_text.set("Status: ESTOP!")
            except Exception:
                logger.exception("EStop error")

    def _on_reset(self):
        if self.ctrl:
            try:
                self.ctrl.reset()
                self.btn_estop.config(state="normal")
                self.btn_offset.configure(state="normal", bg="tomato")
                self.status_text.set("Status: đã reset")
            except Exception:
                logger.exception("Reset error")

    def _on_home(self):
        """Đưa cả 3 motor về gốc index, sau đó phải set_offset lại trước khi Move."""
        if self.ctrl:
            try:
                if hasattr(self.ctrl, "go_home"):
                    self.ctrl.go_home()
                    # Sau home, vị trí encoder reset → isOffset phải = False
                    # để GUI buộc user bấm Set Offset trước khi Move.
                    if hasattr(self.ctrl, "isOffset"):
                        self.ctrl.isOffset = False
                    self.btn_offset.configure(state="normal", bg="tomato")
                elif hasattr(self.ctrl, "set_both_targets"):
                    self.ctrl.set_both_targets(0.0, 0.0, 0.0)
                self.status_text.set("Status: homing về gốc index — bấm Set Offset sau khi xong")
            except Exception:
                logger.exception("Home error")
                messagebox.showerror("Lỗi", "Không thể chạy Home")

    def _on_mode_tog(self):
        if not self.ctrl:
            return
        try:
            state = self.ctrl.get_state()
            if state != CLOSED_LOOP_CONTROL:
                # Currently NOT in closed loop (IDLE or None) → Enter closed loop
                self.ctrl.enter_closed_loop()
                self.btn_mode.config(text="CLOSE ✓", bg="yellow")
            else:
                # Currently in closed loop → Return to IDLE
                self.ctrl.return_IDLE()
                self.btn_mode.config(text="IDLE", bg="lightgreen")
        except Exception:
            logger.exception("Mode toggle error")

    def _apply_bandwidth(self):
        """Tính Kp/Kd từ Control Bandwidth (zeta=1)."""
        try:
            bw_str = self.control_panel[self._ctrl_index["Control bandwidth"]][1].get().strip()
            if not bw_str:
                return
            omega_n = float(bw_str)
            Kp_cal  = omega_n ** 2
            Kd_cal  = 2.0 * omega_n

            for i, key in enumerate(["Kp0", "Kp1", "Kp2"]):
                self.control_panel[self._ctrl_index[key]][1].set(f"{Kp_cal:.2f}")
            for i, key in enumerate(["Kd0", "Kd1", "Kd2"]):
                self.control_panel[self._ctrl_index[key]][1].set(f"{Kd_cal:.2f}")
            self.status_text.set(f"Status: Kp={Kp_cal:.2f}, Kd={Kd_cal:.2f}")
        except ValueError:
            messagebox.showerror("Lỗi", "Nhập số hợp lệ vào Control bandwidth!")

    def _on_optimize_pso(self):
        """Tối ưu Kp/Kd bằng PSO trên quỹ đạo/đáp ứng hiện có."""
        if not self.ctrl:
            messagebox.showwarning("Chưa kết nối", "Hãy kết nối controller trước khi tối ưu.")
            return
        if optimize_kp_kd is None:
            messagebox.showerror("Thiếu module", "Không thể import bộ tối ưu PSO.")
            return

        try:
            bw_str = self.control_panel[self._ctrl_index["Control bandwidth"]][1].get().strip()
            omega_n = float(bw_str) if bw_str else 10.0
            ps = getattr(self.ctrl, "pos_set", 0.0)
            if isinstance(ps, (list, tuple)):
                setpoint = max(abs(ps[0]), abs(ps[1]), 1.0)
            else:
                setpoint = max(abs(ps), 1.0)

            identified = None

            result = optimize_kp_kd(setpoint=setpoint, identified=identified)

            for key in ["Kp0", "Kp1", "Kp2"]:
                self.control_panel[self._ctrl_index[key]][1].set(f"{result.kp:.2f}")
            for key in ["Kd0", "Kd1", "Kd2"]:
                self.control_panel[self._ctrl_index[key]][1].set(f"{result.kd:.2f}")
            try:
                # TWAI signature mới: 11 params
                #   (p0, p1, p2, kp0, kp1, kp2, kd0, kd1, kd2, ctrl_bw, enc_bw)
                # Fallback cho ODriveThread cũ (6 params).
                if isinstance(self.ctrl, twai_controller.TWAIController):
                    ps = self.ctrl.pos_set
                    if isinstance(ps, (list, tuple)):
                        p0, p1, p2 = ps[0], ps[1], ps[2]
                    else:
                        p0 = p1 = p2 = float(ps)
                    self.ctrl.update_ctrlElms(
                        p0, p1, p2,
                        result.kp, result.kp, result.kp,
                        result.kd, result.kd, result.kd,
                        omega_n, self.ctrl.enc_bandwidth,
                        self.ctrl.max_vel,
                    )
                else:
                    target = self.ctrl.pos_set[0] if isinstance(self.ctrl.pos_set, (list, tuple)) else self.ctrl.pos_set
                    self.ctrl.update_ctrlElms(target, self.ctrl.max_vel, result.kp, result.kd, omega_n, self.ctrl.enc_bandwidth)
            except Exception:
                logger.exception("PSO update_ctrlElms failed")

            self.status_text.set(f"Status: PSO Kp={result.kp:.2f}, Kd={result.kd:.2f}, J={result.fitness:.4f}")
            messagebox.showinfo(
                "PSO tối ưu xong",
                f"Kp = {result.kp:.3f}\nKd = {result.kd:.3f}\nFitness = {result.fitness:.6f}"
            )
        except Exception:
            logger.exception("PSO optimize hook error")
            messagebox.showerror("Lỗi", "Không thể tối ưu PSO")

    def _sync_trajectory_mode(self):
        if not self.ctrl:
            return
        mode = (self.trajectory_var.get() or "Spline").strip().lower()
        valid_modes = ("spline", "quintic", "cubic", "trapezoidal")
        if mode not in valid_modes:
            mode = "spline"
            self.trajectory_var.set("Spline")
        self.ctrl.trajectory_mode = mode

    def _on_trajectory_mode_changed(self, _event=None):
        self._sync_trajectory_mode()
        if self.ctrl:
            self.status_text.set(f"Status: Trajectory = {self.trajectory_var.get()}")

    def _apply_trajectory_mode(self):
        self._sync_trajectory_mode()
        if self.ctrl:
            mode = self.trajectory_var.get()
            print(f"[GUI] Apply Trajectory clicked -> mode={mode}")
            # Re-init trajectory objects khi đổi mode (tránh lẫn state giữa các loại)
            try:
                with self.ctrl.data_lock:
                    self.ctrl._init_trajectories()
            except Exception as e:
                print(f"[GUI] Re-init traj[] failed: {e}")
            try:
                t0 = float(self.control_panel[0][1].get().strip() or "0")
                t1 = float(self.control_panel[1][1].get().strip() or "0")
                t2 = float(self.control_panel[2][1].get().strip() or "0")
            except (ValueError, IndexError, tk.TclError):
                t0 = t1 = t2 = 0.0
            kp0 = float(self.control_panel[self._ctrl_index["Kp0"]][1].get().strip() or "0")
            kp1 = float(self.control_panel[self._ctrl_index["Kp1"]][1].get().strip() or "0")
            kp2 = float(self.control_panel[self._ctrl_index["Kp2"]][1].get().strip() or "0")
            kd0 = float(self.control_panel[self._ctrl_index["Kd0"]][1].get().strip() or "0")
            kd1 = float(self.control_panel[self._ctrl_index["Kd1"]][1].get().strip() or "0")
            kd2 = float(self.control_panel[self._ctrl_index["Kd2"]][1].get().strip() or "0")
            self._apply_axis_locks()
            # Entry world frame → trừ model_home_deg trước khi gửi relative target
            try:
                mh = self.ctrl.get_model_home()
            except Exception:
                mh = (0.0, 0.0, 0.0)
            rel_targets = [t0 - mh[0], t1 - mh[1], t2 - mh[2]]
            self.ctrl.update_ctrlElms(
                rel_targets[0], rel_targets[1], rel_targets[2],
                kp0, kp1, kp2,
                kd0, kd1, kd2,
                self.ctrl.ctrl_bandwidth,
                self.ctrl.enc_bandwidth,
                self.ctrl.max_vel,
            )
            if callable(getattr(self.ctrl, "apply_gui_targets_deg", None)):
                self._last_running_targets = None
                self.ctrl.apply_gui_targets_deg(rel_targets[0], rel_targets[1], rel_targets[2])
            self.status_text.set(f"Status: Applied Trajectory = {mode}")

    def _apply_axis_locks(self):
        if self.ctrl and hasattr(self.ctrl, "locked_axes"):
            self.ctrl.locked_axes = [bool(v.get()) for v in self.axis_lock_vars]
            self.ctrl.friction_only_mode = False
            if hasattr(self.ctrl, "set_locked_axes"):
                locks = [bool(v.get()) for v in self.axis_lock_vars]
                self.ctrl.set_locked_axes(locks[0], locks[1], locks[2])

    def _on_move(self):
        """Chỉ arm motion khi người dùng bấm Move."""
        if not self.ctrl:
            return
        try:
            elms = [float(v.get().strip() or "0") for _, v in self.control_panel]
            p0, p1 = elms[0], elms[1]

            self._apply_axis_locks()
            # Entry hiển thị hệ world (deg). update_ctrlElms/_set_target_state dùng hệ relative.
            # Trừ model_home_deg để target relative đúng → spline chạy tới đúng vị trí world.
            try:
                mh = self.ctrl.get_model_home()
            except Exception:
                mh = (0.0, 0.0, 0.0)
            rel_targets = [elms[i] - mh[i] for i in range(3)]
            print(f"[GUI] Run CTC Motion targets world=({elms[0]:.6f}, {elms[1]:.6f}, {elms[2]:.6f}) rel=({rel_targets[0]:.6f}, {rel_targets[1]:.6f}, {rel_targets[2]:.6f}), locks = {[bool(v.get()) for v in self.axis_lock_vars]}, traj = {self.trajectory_var.get()}")
            print(f"[GUI] Kp/Kd axes = Kp({self.control_panel[self._ctrl_index['Kp0']][1].get()}, {self.control_panel[self._ctrl_index['Kp1']][1].get()}, {self.control_panel[self._ctrl_index['Kp2']][1].get()}) Kd({self.control_panel[self._ctrl_index['Kd0']][1].get()}, {self.control_panel[self._ctrl_index['Kd1']][1].get()}, {self.control_panel[self._ctrl_index['Kd2']][1].get()})")

            kp0 = float(self.control_panel[self._ctrl_index["Kp0"]][1].get().strip() or "0")
            kp1 = float(self.control_panel[self._ctrl_index["Kp1"]][1].get().strip() or "0")
            kp2 = float(self.control_panel[self._ctrl_index["Kp2"]][1].get().strip() or "0")
            kd0 = float(self.control_panel[self._ctrl_index["Kd0"]][1].get().strip() or "0")
            kd1 = float(self.control_panel[self._ctrl_index["Kd1"]][1].get().strip() or "0")
            kd2 = float(self.control_panel[self._ctrl_index["Kd2"]][1].get().strip() or "0")
            max_vel = float(
                self.control_panel[self._ctrl_index["Max Trajectory Velocity"]][1].get().strip()
                or "60"
            )

            # Cập nhật target trước, rồi arm motion ở cuối để tránh backend
            # vô tình bị disarm bởi update_ctrlElms()/set_both_targets().
            self.ctrl.update_ctrlElms(
                rel_targets[0], rel_targets[1], rel_targets[2],
                kp0, kp1, kp2,
                kd0, kd1, kd2,
                self.ctrl.ctrl_bandwidth,
                self.ctrl.enc_bandwidth,
                max_vel,
            )

            if hasattr(self.ctrl, "apply_gui_targets_deg"):
                self.ctrl.apply_gui_targets_deg(rel_targets[0], rel_targets[1], rel_targets[2])

            if hasattr(self.ctrl, "motion_armed"):
                self.ctrl.motion_armed = True
            self._last_running_targets = None

            backend = self.backend_var.get()
            self.status_text.set(f"Status: {backend} M0={p0:.2f}°, M1={p1:.2f}°")
        except Exception:
            logger.exception("Move error")
            messagebox.showerror("Lỗi", "Không thể gửi lệnh Move")

    def _on_send_params(self):
        """Gửi load parameters."""
        if not self.ctrl:
            return
        try:
            params = []
            for _, v, key in self.param_panel:
                value = float(v.get().strip() or "0")
                if key in self._load_input_cfg:
                    min_v, max_v, _ = self._load_input_cfg[key]
                    if not (min_v <= value <= max_v):
                        raise ValueError(
                            f"{key} phải nằm trong [{min_v:.3f}, {max_v:.3f}]"
                        )
                params.append(value)
            self.ctrl.update_loadParms(*params)
            self.status_text.set("Status: parameters đã gửi")
        except ValueError as e:
            messagebox.showerror("Lỗi nhập liệu", str(e))
        except Exception:
            logger.exception("Send params error")
            messagebox.showerror("Lỗi", "Không thể gửi Parameters")

    def _on_send_prismatic(self):
        """Gửi prismatic positions tới ESP32."""
        if not self.ctrl:
            return
        try:
            hip_mm = float(self._prism_vars["Hip prismatic (mm)"].get())
            knee_mm = float(self._prism_vars["Knee prismatic (mm)"].get())
            
            # Clamp to valid range
            hip_mm = max(350.0, min(450.0, hip_mm))
            knee_mm = max(350.0, min(450.0, knee_mm))
            
            # Check if set_prismatic method exists
            if hasattr(self.ctrl, 'set_prismatic'):
                self.ctrl.set_prismatic(hip_mm, knee_mm)
                self.status_text.set(f"Status: Prismatic gửi - Hip={hip_mm:.1f}mm, Knee={knee_mm:.1f}mm")
            else:
                messagebox.showwarning("Cảnh báo", "Controller không hỗ trợ prismatic")
        except ValueError as e:
            messagebox.showerror("Lỗi nhập liệu", str(e))
        except Exception:
            logger.exception("Send prismatic error")
            messagebox.showerror("Lỗi", "Không thể gửi Prismatic")

    def _on_send_delta_x(self):
        """Cập nhật khoảng tịnh tiến COM (delta_x1, delta_x2) cho CTC."""
        if not self.ctrl:
            return
        try:
            delta_x1 = float(self._delta_x1_var.get())
            delta_x2 = float(self._delta_x2_var.get())

            # Send DX command via serial
            cmd = f"DX:{delta_x1:.4f},{delta_x2:.4f}"
            self.ctrl._send_simple_cmd(cmd)
            self.status_text.set(f"Status: COM updated - Hip Δx={delta_x1:.4f}m, Knee Δx={delta_x2:.4f}m")
        except ValueError as e:
            messagebox.showerror("Lỗi nhập liệu", str(e))
        except AttributeError:
            # Fallback: try set_delta_x method if serial_write not available
            if hasattr(self.ctrl, 'set_delta_x'):
                self.ctrl.set_delta_x(delta_x1, delta_x2)
                self.status_text.set(f"Status: COM updated - Hip Δx={delta_x1:.4f}m, Knee Δx={delta_x2:.4f}m")
            else:
                messagebox.showerror("Lỗi", "Controller không hỗ trợ")
        except Exception:
            logger.exception("Send delta_x error")
            messagebox.showerror("Lỗi", "Không thể cập nhật delta_x")

    def _on_cylinder(self, joint_idx: int, direction: int):
        """Điều khiển cylinder: joint_idx=0(hip),1(knee), direction=1(kéo dài),-1(thu ngắn)."""
        if not self.ctrl:
            return
        try:
            # Gửi lệnh cylinder đến ESP32
            # direction > 0: kéo dài (cộng chiều dài)
            # direction < 0: thu ngắn (trừ chiều dài)
            if hasattr(self.ctrl, 'control_cylinder'):
                self.ctrl.control_cylinder(joint_idx, direction)
                dir_name = "kéo dài" if direction > 0 else "thu ngắn"
                joint_name = "Hip" if joint_idx == 0 else "Knee"
                self.status_text.set(f"Status: Cylinder {joint_name} đang {dir_name}...")
            else:
                # Mock nếu chưa có method - cập nhật laser giả lập
                pass
        except Exception:
            logger.exception("Cylinder control error")

    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _set_entry(widget, value: str):
        widget.config(state="normal")
        widget.delete(0, tk.END)
        widget.insert(0, value)
        widget.config(state="readonly")

    @staticmethod
    def _estimate_vel(times: list[float], pos_vals: list[float]) -> list[float]:
        """Ước lượng gia tốc (deg/s²) từ chuỗi vị trí theo thời gian."""
        n = len(times)
        if n < 3:
            return [0.0] * n
        acc = [0.0] * n
        for i in range(1, n - 1):
            dt1 = times[i] - times[i - 1]
            dt2 = times[i + 1] - times[i]
            if dt1 > 1e-6 and dt2 > 1e-6:
                v1 = (pos_vals[i] - pos_vals[i - 1]) / dt1
                v2 = (pos_vals[i + 1] - pos_vals[i]) / dt2
                acc[i] = (v2 - v1) / ((dt1 + dt2) * 0.5)
        return acc

    # ════════════════════════════════════════════════════════════════════════
    # Periodic update (50ms)
    # ════════════════════════════════════════════════════════════════════════

    def _update(self):
        try:
            ctrl = self.ctrl

            # ── Update status label ───────────────────────────────────────
            if ctrl is None:
                self.status_label.config(text="Chưa kết nối", background="lightgrey")
                self.btn_move.configure(state="disabled")
                self.btn_send_param.configure(state="disabled")
                self.btn_optimize_pso.configure(state="disabled")
            else:
                connected   = bool(getattr(ctrl, "connected",          False))
                ready       = bool(getattr(ctrl, "esp32_ready",        False))
                closed_loop = bool(getattr(ctrl, "closed_loop_control", False))
                is_offset   = bool(getattr(ctrl, "isOffset",           False))
                estop       = bool(getattr(ctrl, "_estop_event",
                                           threading.Event()).is_set())
                error       = bool(getattr(ctrl, "error",              False))
                msg         = getattr(ctrl, "status_message",          "")
                backend     = self.backend_var.get()

                # Status colour
                motion_armed = bool(getattr(ctrl, "motion_armed", False))
                if estop:
                    self.status_label.config(text=f"{backend} | ESTOP", background="red")
                elif error:
                    self.status_label.config(text=f"{backend} | ERROR", background="orange")
                elif closed_loop and motion_armed:
                    self.status_label.config(text=f"{backend} | RUNNING", background="yellow")
                elif closed_loop:
                    self.status_label.config(text=f"{backend} | TORQUE READY", background="lightgreen")
                elif ready:
                    self.status_label.config(text=f"{backend} | READY", background="lightgreen")
                elif connected:
                    self.status_label.config(text=f"{backend} | Đang chờ...", background="lightyellow")
                else:
                    self.status_label.config(text=f"{backend} | Disconnected", background="lightgrey")

                # Mode button text
                if closed_loop:
                    self.btn_mode.config(text="→ IDLE", bg="yellow")
                else:
                    self.btn_mode.config(text="Enable Torque", bg="lightgreen")

                # Enable/disable buttons
                move_ok  = is_offset and closed_loop and not estop
                param_ok = connected and not estop
                self.btn_move.configure(      state="normal" if move_ok  else "disabled")
                self.btn_home.configure(state="normal" if connected and not estop else "disabled")
                self.btn_send_param.configure(state="normal" if param_ok else "disabled")
                self.btn_send_prismatic.configure(state="normal" if param_ok else "disabled")
                self.btn_send_delta.configure(state="normal" if param_ok else "disabled")
                self.btn_optimize_pso.configure(state="normal" if connected and not estop else "disabled")

                # Enable cylinder buttons when connected
                cyl_state = "normal" if connected and not estop else "disabled"
                self._btn_hip_plus.configure(state=cyl_state)
                self._btn_hip_minus.configure(state=cyl_state)
                self._btn_knee_plus.configure(state=cyl_state)
                self._btn_knee_minus.configure(state=cyl_state)

                # Offset button
                if is_offset:
                    self.btn_offset.configure(state="disabled", bg="lightgreen")
                else:
                    self.btn_offset.configure(state="normal",   bg="tomato")

                # TWAI: Target Motor KHÔNG tự gửi khi đổi Entry — chỉ gửi khi bấm ▶ Run CTC Motion.
                # Tự động gửi mỗi tick gây hạ moment không mong muốn (đặc biệt khớp 1).
                # self._last_running_targets KHÔNG còn dùng; giữ lại cho compat nếu nơi khác cần.

                # ── Update status bar ─────────────────────────────────────
                if msg:
                    self.status_text.set(f"Status: {msg}")
                elif closed_loop and motion_armed and callable(getattr(ctrl, "apply_gui_targets_deg", None)):
                    self.status_text.set(
                        "Status: RUNNING — đổi Target Motor chỉ áp dụng sau khi bấm ▶ Run CTC Motion."
                    )
                elif closed_loop and callable(getattr(ctrl, "apply_gui_targets_deg", None)):
                    self.status_text.set(
                        "Status: Nhập Target Motor 0/1/2, rồi bấm ▶ Run CTC Motion để gửi lệnh."
                    )
                elif closed_loop:
                    self.status_text.set("Status: torque control running")

                # ── Update position boxes ─────────────────────────────────
                if connected:
                    p0, p1, p2 = ctrl.get_pos()
                    self._set_entry(self.entry_pos0, f"{p0:.3f}")
                    self._set_entry(self.entry_pos1, f"{p1:.3f}")
                    self._set_entry(self.entry_pos2, f"{p2:.3f}")

                    if hasattr(ctrl, "get_setpoints"):
                        ps0, ps1, ps2 = ctrl.get_setpoints()
                    else:
                        with ctrl.data_lock:
                            ps = getattr(ctrl, "pos_set", [0.0, 0.0, 0.0])
                            if isinstance(ps, (list, tuple)):
                                ps0, ps1, ps2 = ps[0], ps[1], ps[2]
                            else:
                                ps0, ps1, ps2 = ps, 0.0, 0.0

                # ── Update plot ───────────────────────────────────────────
                if self.plotting and ctrl and connected:
                    data = ctrl.get_data()
                    if data:
                        times     = [d[0] for d in data]
                        # Data tuple MỚI (19 field, controller đã compute acc bằng SavGol):
                        #   d[0]   = timestamp
                        #   d[1..3]= pos[0..2],  d[4..6]= pos_set[0..2]
                        #   d[7..9]= vel[0..2],  d[10..12]= vel_set[0..2]
                        #   d[13..15]= acc[0..2] (deg/s^2, pre-computed)
                        #   d[16..18]= tau_out[0..2] (Nm, motor-side)
                        # GUI chỉ cần unzip + set_data, KHÔNG tính acc ở đây nữa
                        # (giống ODESC: chỉ vẽ, không compute).
                        n = len(times)
                        nd = len(data[0])
                        # Vectorize bằng numpy thay vì list comprehension (nhanh hơn nhiều).
                        _arr = np.asarray(data, dtype=float)
                        # pos & setpoint cộng model_home_deg để cùng hệ world.
                        try:
                            mh = ctrl.get_model_home()
                        except Exception:
                            mh = (0.0, 0.0, 0.0)
                        pos0_vals = (_arr[:, 1] + mh[0]).tolist()
                        pos1_vals = (_arr[:, 2] + mh[1]).tolist()
                        pos2_vals = (_arr[:, 3] + mh[2]).tolist()
                        set0_vals = (_arr[:, 4] + mh[0]).tolist()
                        set1_vals = (_arr[:, 5] + mh[1]).tolist()
                        set2_vals = (_arr[:, 6] + mh[2]).tolist()
                        err0_vals = (_arr[:, 4] - _arr[:, 1]).tolist()
                        err1_vals = (_arr[:, 5] - _arr[:, 2]).tolist()
                        err2_vals = (_arr[:, 6] - _arr[:, 3]).tolist()
                        vel0_vals = _arr[:, 7].tolist()  if nd >= 10 else []
                        vel1_vals = _arr[:, 8].tolist()  if nd >= 11 else []
                        vel2_vals = _arr[:, 9].tolist()  if nd >= 12 else []
                        vel0_set  = _arr[:, 10].tolist() if nd >= 13 else []
                        vel1_set  = _arr[:, 11].tolist() if nd >= 14 else []
                        vel2_set  = _arr[:, 12].tolist() if nd >= 15 else []
                        # Acc: đọc thẳng từ controller (đã tính sẵn bằng SavGol).
                        acc0_vals = _arr[:, 13].tolist() if nd >= 16 else []
                        acc1_vals = _arr[:, 14].tolist() if nd >= 17 else []
                        acc2_vals = _arr[:, 15].tolist() if nd >= 18 else []
                        # acc_set: chưa push vào data tuple (gọn 19 field). Để trống
                        # cho ref line; nếu cần ref acc, mở rộng tuple trong controller.
                        acc0_set = acc1_set = acc2_set = []
                        # Torque DK: d[16..18] (Nm motor-side, đã qua motor_sign).
                        tau0_vals = _arr[:, 16].tolist() if nd >= 19 else []
                        tau1_vals = _arr[:, 17].tolist() if nd >= 20 else []
                        tau2_vals = _arr[:, 18].tolist() if nd >= 21 else []
                        # CTC raw torque: không còn trong data tuple (đã gọn bỏ).
                        tauctc0_vals = tauctc1_vals = tauctc2_vals = []

                        t0 = times[0] if self._last_t0 is None else self._last_t0
                        if self._last_t0 is None or (times[-1] - t0) > 30.0:
                            t0 = times[0]
                            self._last_t0 = t0
                        t_rel = [t - t0 for t in times]

                        # Helper: cập nhật line với guard tránh matplotlib crash khi
                        # set_data(t_rel, []) (relim không broadcast được khi x≠y).
                        def _safe_set(line, x, y):
                            if x is None or y is None or len(x) != len(y):
                                # Reset về cả hai rỗng (kích thước khớp = 0).
                                line.set_data([], [])
                            else:
                                line.set_data(x, y)

                        # ── Reset trước tất cả line để tránh carry-over shape từ tick trước
                        # (matplotlib Line2D._xy không được reset sạch bởi set_data([], [])
                        # trong một số phiên bản — phòng xa bằng cách clear hết).
                        _safe_set(self._line_pos0,   t_rel, pos0_vals)
                        _safe_set(self._line_pos1,   t_rel, pos1_vals)
                        _safe_set(self._line_pos2,   t_rel, pos2_vals)
                        _safe_set(self._line_pos0_set, t_rel, set0_vals)
                        _safe_set(self._line_pos1_set, t_rel, set1_vals)
                        _safe_set(self._line_pos2_set, t_rel, set2_vals)
                        _safe_set(self._line_vel0,   t_rel, vel0_vals)
                        _safe_set(self._line_vel1,   t_rel, vel1_vals)
                        _safe_set(self._line_vel2,   t_rel, vel2_vals)
                        _safe_set(self._line_vel0_set, t_rel, vel0_set)
                        _safe_set(self._line_vel1_set, t_rel, vel1_set)
                        _safe_set(self._line_vel2_set, t_rel, vel2_set)
                        _safe_set(self._line_acc0,   t_rel, acc0_vals)
                        _safe_set(self._line_acc1,   t_rel, acc1_vals)
                        _safe_set(self._line_acc2,   t_rel, acc2_vals)
                        _safe_set(self._line_acc0_set, t_rel, acc0_set)
                        _safe_set(self._line_acc1_set, t_rel, acc1_set)
                        _safe_set(self._line_acc2_set, t_rel, acc2_set)
                        _safe_set(self._line_tau0,   t_rel, tau0_vals)
                        _safe_set(self._line_tau1,   t_rel, tau1_vals)
                        _safe_set(self._line_tau2,   t_rel, tau2_vals)
                        _safe_set(self._line_tauctc0, t_rel, tauctc0_vals)
                        _safe_set(self._line_tauctc1, t_rel, tauctc1_vals)
                        _safe_set(self._line_tauctc2, t_rel, tauctc2_vals)

                        # Wrap relim trong try/except — một số matplotlib version
                        # crash nếu line có x/y mismatch length carry-over từ tick trước.
                        try:
                            self.ax_pos.relim(); self.ax_pos.autoscale_view()
                        except Exception:
                            pass
                        try:
                            self.ax_vel.relim(); self.ax_vel.autoscale_view()
                        except Exception:
                            pass
                        try:
                            self.ax_acc.relim(); self.ax_acc.autoscale_view()
                        except Exception:
                            pass
                        try:
                            self.ax_tau.relim(); self.ax_tau.autoscale_view()
                        except Exception:
                            pass
                        self.canvas.draw_idle()

        except Exception:
            logger.exception("GUI update error")

        self.after(UPDATE_INTERVAL_MS, self._update)

    # ════════════════════════════════════════════════════════════════════════
    # Close
    # ════════════════════════════════════════════════════════════════════════

    def _on_close(self):
        if messagebox.askokcancel("Thoát", "Bạn có muốn thoát không?"):
            try:
                if self.ctrl:
                    self.ctrl.stop()
                    self.ctrl.join(timeout=2.0)
            except Exception:
                logger.exception("Shutdown error")
            finally:
                self.destroy()


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = ControlGUI()
    app.mainloop()