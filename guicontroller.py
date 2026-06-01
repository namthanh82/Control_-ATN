import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import logging
from collections import deque

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
    from data_logger import snapshot_from_controller
except Exception:
    snapshot_from_controller = None

try:
    from system_identifier import estimate_first_order_model
except Exception:
    estimate_first_order_model = None

# ── State aliases ────────────────────────────────────────────────────────────
IDLE             = twai_controller.IDLE
CLOSE_LOOP_CONTROL = twai_controller.CLOSED_LOOP_CONTROL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CTCControlGUI")

UPDATE_INTERVAL_MS = 50   # 20 Hz GUI update


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

        # ── Params mặc định cho control panel ───────────────────────────
        self._default_ctrl = {
            "Target Motor 0": (0.0,   "deg"),
            "Target Motor 1": (0.0,   "deg"),
            "Target Motor 2": (0.0,   "deg"),
            "Kp0":            (3.0,   None),
            "Kp1":            (3.0,   None),
            "Kp2":            (3.0,   None),
            "Kd0":            (1.0,   None),
            "Kd1":            (1.0,   None),
            "Kd2":            (1.0,   None),
            "Control bandwidth": (1200.0, None),
            "Encoder bandwidth": (50.0,   None),
        }
        self._ctrl_index = {}
        self.trajectory_var = tk.StringVar(value="Spline")
        self._default_load = {
            "External load":   (0.0,   "kg"),
            "Load position":   (0.0,   "m"),
            "Coulomb friction":(0.0,   "Nm"),
            "Viscous friction":(0.0,  "Nm/(rad/s)"), #0.00276
            "Torque limit":    (0.1,  "Nm"),
        }
        # Cho phép điều chỉnh dải nhập load ngay tại GUI.
        # Mỗi tham số: (min, max, step)
        self._load_input_cfg = {
            "External load":    (0.0, 0.0, 0.0),
            "Load position":    (0.0, 0.0, 0.0),
            "Coulomb friction": (0.0, 0.0, 0.0),
            "Viscous friction": (0.0, 0.0, 0.0),
            "Torque limit":     (0.2, 0.2, 0.2),
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

        right = ttk.Frame(main, width=380)
        right.pack(side=tk.RIGHT, fill=tk.BOTH)
        right.pack_propagate(False)

        # ── Plot ─────────────────────────────────────────────────────────
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax_pos = self.fig.add_subplot(311)
        self.ax_vel = self.fig.add_subplot(312)
        self.ax_acc = self.fig.add_subplot(313)
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Lines
        self._line_pos0,     = self.ax_pos.plot([], [], label="Motor 0 (deg)", color="#2196F3")
        self._line_pos1,     = self.ax_pos.plot([], [], label="Motor 1 (deg)", color="#FF9800")
        self._line_pos2,     = self.ax_pos.plot([], [], label="Motor 2 (deg)", color="#4CAF50")
        self._line_pos0_set, = self.ax_pos.plot([], [], label="M0 Setpoint",   color="#2196F3",
                                                linestyle="--", alpha=0.6)
        self._line_pos1_set, = self.ax_pos.plot([], [], label="M1 Setpoint",   color="#FF9800",
                                                linestyle="--", alpha=0.6)
        self._line_pos2_set, = self.ax_pos.plot([], [], label="M2 Setpoint",   color="#4CAF50",
                                                linestyle="--", alpha=0.6)
        # Error lines in second subplot
        self._line_err0, = self.ax_vel.plot([], [], label="Error M0 (deg)", color="#2196F3")
        self._line_err1, = self.ax_vel.plot([], [], label="Error M1 (deg)", color="#FF9800")
        self._line_err2, = self.ax_vel.plot([], [], label="Error M2 (deg)", color="#4CAF50")

        # Acceleration lines in third subplot
        self._line_acc0, = self.ax_acc.plot([], [], label="Accel M0", color="#2196F3")
        self._line_acc1, = self.ax_acc.plot([], [], label="Accel M1", color="#FF9800")
        self._line_acc2, = self.ax_acc.plot([], [], label="Accel M2", color="#4CAF50")
        self._line_acc0_set, = self.ax_acc.plot([], [], label="Accel M0 Set", color="#2196F3", linestyle="--", alpha=0.6)
        self._line_acc1_set, = self.ax_acc.plot([], [], label="Accel M1 Set", color="#FF9800", linestyle="--", alpha=0.6)
        self._line_acc2_set, = self.ax_acc.plot([], [], label="Accel M2 Set", color="#4CAF50", linestyle="--", alpha=0.6)

        self.ax_pos.set_ylabel("Position (deg)")
        self.ax_pos.grid(True);  self.ax_pos.legend(loc="upper right", fontsize=7)
        self.ax_vel.set_ylabel("Error (deg)")
        self.ax_vel.grid(True);  self.ax_vel.legend(loc="upper right", fontsize=7)
        self.ax_acc.set_ylabel("Acceleration")
        self.ax_acc.set_xlabel("Time (s)")
        self.ax_acc.grid(True);  self.ax_acc.legend(loc="upper right", fontsize=7)

        # ── Action buttons ────────────────────────────────────────────────
        top_right = ttk.Frame(right, padding=6)
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
        ctrl_frame = ttk.LabelFrame(right, text="Điều Khiển", padding=8)
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
            values=["Spline", "Quintic"],
            state="readonly",
            width=10,
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
        param_frame = ttk.LabelFrame(right, text="Parameters", padding=8)
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

        # ── Error display ─────────────────────────────────────────────────
        err_frame = ttk.LabelFrame(right, text="Sai số", padding=8)
        err_frame.pack(padx=6, pady=6, fill=tk.X)

        ttk.Label(err_frame, text="Error M0 (deg):").grid(row=0, column=0, sticky=tk.W)
        self.entry_err0 = ttk.Entry(err_frame, width=12, state="readonly")
        self.entry_err0.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(err_frame, text="Error M1 (deg):").grid(row=1, column=0, sticky=tk.W)
        self.entry_err1 = ttk.Entry(err_frame, width=12, state="readonly")
        self.entry_err1.grid(row=1, column=1, padx=4, pady=2)

        ttk.Label(err_frame, text="Error M2 (deg):").grid(row=2, column=0, sticky=tk.W)
        self.entry_err2 = ttk.Entry(err_frame, width=12, state="readonly")
        self.entry_err2.grid(row=2, column=1, padx=4, pady=2)

        # ── Status bar ────────────────────────────────────────────────────
        status_frame = ttk.Frame(right, padding=6)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = tk.StringVar(value="Status: chưa kết nối")
        ttk.Label(status_frame, textvariable=self.status_text,
                  relief=tk.RIDGE).pack(fill=tk.X)

        self._ensure_error_entries()

    # ════════════════════════════════════════════════════════════════════════
    # Connection management
    # ════════════════════════════════════════════════════════════════════════

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
        if self.ctrl:
            try:
                if hasattr(self.ctrl, "go_home"):
                    self.ctrl.go_home()
                elif hasattr(self.ctrl, "set_both_targets"):
                    self.ctrl.set_both_targets(0.0, 0.0, 0.0)
                self.status_text.set("Status: homing về gốc index")
            except Exception:
                logger.exception("Home error")
                messagebox.showerror("Lỗi", "Không thể chạy Home")

    def _on_mode_tog(self):
        if not self.ctrl:
            return
        try:
            state = self.ctrl.get_state()
            if state == IDLE or state is None:
                self.ctrl.enter_closed_loop()
                self.btn_mode.config(text="→ IDLE", bg="yellow")
            else:
                self.ctrl.return_IDLE()
                self.btn_mode.config(text="Enable Torque", bg="lightgreen")
        except Exception:
            logger.exception("Mode toggle error")

    def _apply_bandwidth(self):
        """Tính Kp/Kd từ Control Bandwidth (zeta=1)."""
        try:
            keys = list(self._default_ctrl.keys())
            idx_bw = keys.index("Control bandwidth")
            idx_kp = keys.index("Kp")
            idx_kd = keys.index("Kd")

            bw_str = self.control_panel[idx_bw][1].get().strip()
            if not bw_str:
                return
            omega_n = float(bw_str)
            Kp_cal  = omega_n ** 2
            Kd_cal  = 2.0 * omega_n

            self.control_panel[idx_kp][1].set(f"{Kp_cal:.2f}")
            self.control_panel[idx_kd][1].set(f"{Kd_cal:.2f}")
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
            keys = list(self._default_ctrl.keys())
            idx_kp = keys.index("Kp")
            idx_kd = keys.index("Kd")
            idx_bw = keys.index("Control bandwidth")

            bw_str = self.control_panel[idx_bw][1].get().strip()
            omega_n = float(bw_str) if bw_str else 10.0
            ps = getattr(self.ctrl, "pos_set", 0.0)
            if isinstance(ps, (list, tuple)):
                setpoint = max(abs(ps[0]), abs(ps[1]), 1.0)
            else:
                setpoint = max(abs(ps), 1.0)

            identified = None
            if snapshot_from_controller is not None and estimate_first_order_model is not None:
                snap = snapshot_from_controller(self.ctrl)
                if snap["time"] and snap["pos0"]:
                    identified = estimate_first_order_model(snap["time"], snap["pos0"], setpoint)

            result = optimize_kp_kd(setpoint=setpoint, identified=identified)

            self.control_panel[idx_kp][1].set(f"{result.kp:.2f}")
            self.control_panel[idx_kd][1].set(f"{result.kd:.2f}")
            try:
                if isinstance(self.ctrl, twai_controller.TWAIController):
                    self.ctrl.update_ctrlElms(self.ctrl.pos_set[0], self.ctrl.pos_set[1], result.kp, result.kd, omega_n, self.ctrl.enc_bandwidth)
                else:
                    target = self.ctrl.pos_set[0] if isinstance(self.ctrl.pos_set, (list, tuple)) else self.ctrl.pos_set
                    self.ctrl.update_ctrlElms(target, self.ctrl.max_vel, result.kp, result.kd, omega_n, self.ctrl.enc_bandwidth)
            except Exception:
                pass

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
        self.ctrl.trajectory_mode = "quintic" if mode == "quintic" else "spline"

    def _on_trajectory_mode_changed(self, _event=None):
        self._sync_trajectory_mode()
        if self.ctrl:
            self.status_text.set(f"Status: Trajectory = {self.trajectory_var.get()}")

    def _apply_trajectory_mode(self):
        self._sync_trajectory_mode()
        if self.ctrl:
            mode = self.trajectory_var.get()
            print(f"[GUI] Apply Trajectory clicked -> mode={mode}")
            print(f"[GUI] Controller trajectory_mode set to {self.ctrl.trajectory_mode}")
            kp0 = float(self.control_panel[self._ctrl_index["Kp0"]][1].get().strip() or "0")
            kp1 = float(self.control_panel[self._ctrl_index["Kp1"]][1].get().strip() or "0")
            kp2 = float(self.control_panel[self._ctrl_index["Kp2"]][1].get().strip() or "0")
            kd0 = float(self.control_panel[self._ctrl_index["Kd0"]][1].get().strip() or "0")
            kd1 = float(self.control_panel[self._ctrl_index["Kd1"]][1].get().strip() or "0")
            kd2 = float(self.control_panel[self._ctrl_index["Kd2"]][1].get().strip() or "0")
            print(
                f"[GUI] Current setpoints pos_set=({self.ctrl.pos_set[0]:.6f}, {self.ctrl.pos_set[1]:.6f}, {self.ctrl.pos_set[2]:.6f}) "
                f"Kp=({kp0:.3f}, {kp1:.3f}, {kp2:.3f}) Kd=({kd0:.3f}, {kd1:.3f}, {kd2:.3f}) bw={self.ctrl.ctrl_bandwidth:.3f} enc_bw={self.ctrl.enc_bandwidth:.3f}"
            )
            self.ctrl.update_ctrlElms(
                self.ctrl.pos_set[0],
                self.ctrl.pos_set[1],
                self.ctrl.pos_set[2],
                kp0, kp1, kp2,
                kd0, kd1, kd2,
                self.ctrl.ctrl_bandwidth,
                self.ctrl.enc_bandwidth,
            )
            self.status_text.set(f"Status: Applied Trajectory = {mode}")

    def _apply_axis_locks(self):
        if self.ctrl and hasattr(self.ctrl, "locked_axes"):
            self.ctrl.locked_axes = [bool(v.get()) for v in self.axis_lock_vars]
            self.ctrl.friction_only_mode = False

    def _on_move(self):
        """Chỉ arm motion khi người dùng bấm Move."""
        if not self.ctrl:
            return
        try:
            elms = [float(v.get().strip() or "0") for _, v in self.control_panel]
            p0, p1 = elms[0], elms[1]

            self._apply_axis_locks()
            print(f"[GUI] Run CTC Motion targets = ({elms[0]:.6f}, {elms[1]:.6f}, {elms[2]:.6f}), locks = {[bool(v.get()) for v in self.axis_lock_vars]}, traj = {self.trajectory_var.get()}")

            kp0 = float(self.control_panel[self._ctrl_index["Kp0"]][1].get().strip() or "0")
            kp1 = float(self.control_panel[self._ctrl_index["Kp1"]][1].get().strip() or "0")
            kp2 = float(self.control_panel[self._ctrl_index["Kp2"]][1].get().strip() or "0")
            kd0 = float(self.control_panel[self._ctrl_index["Kd0"]][1].get().strip() or "0")
            kd1 = float(self.control_panel[self._ctrl_index["Kd1"]][1].get().strip() or "0")
            kd2 = float(self.control_panel[self._ctrl_index["Kd2"]][1].get().strip() or "0")

            # Cập nhật target trước, rồi arm motion ở cuối để tránh backend
            # vô tình bị disarm bởi update_ctrlElms()/set_both_targets().
            self.ctrl.update_ctrlElms(
                elms[0], elms[1], elms[2],
                kp0, kp1, kp2,
                kd0, kd1, kd2,
                self.ctrl.ctrl_bandwidth,
                self.ctrl.enc_bandwidth,
            )

            if hasattr(self.ctrl, "motion_armed"):
                self.ctrl.motion_armed = True

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

    # ════════════════════════════════════════════════════════════════════════
    # Helpers to set entry values
    # ════════════════════════════════════════════════════════════════════════

    def _ensure_error_entries(self):
        if not hasattr(self, "entry_err2"):
            self.entry_err2 = ttk.Entry(self, width=12, state="readonly")

    def _set_entry(self, widget, value: str):
        widget.config(state="normal")
        widget.delete(0, tk.END)
        widget.insert(0, value)
        widget.config(state="readonly")

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
                self.btn_optimize_pso.configure(state="normal" if connected and not estop else "disabled")

                # Offset button
                if is_offset:
                    self.btn_offset.configure(state="disabled", bg="lightgreen")
                else:
                    self.btn_offset.configure(state="normal",   bg="tomato")

                # TWAI: khi RUNNING, đẩi số trong ô Target Motor → pos_set (không cần bấm Run lại)
                if (
                    motion_armed
                    and closed_loop
                    and is_offset
                    and not estop
                    and callable(getattr(ctrl, "apply_gui_targets_deg", None))
                ):
                    try:
                        t0 = float(self.control_panel[0][1].get().strip() or "0")
                        t1 = float(self.control_panel[1][1].get().strip() or "0")
                        t2 = float(self.control_panel[2][1].get().strip() or "0")
                        print(f"[GUI] RUNNING targets -> M0={t0:.3f}, M1={t1:.3f}, M2={t2:.3f}")
                        ctrl.apply_gui_targets_deg(t0, t1, t2)
                    except (ValueError, IndexError, tk.TclError):
                        pass

                # ── Update status bar ─────────────────────────────────────
                if msg:
                    self.status_text.set(f"Status: {msg}")
                elif closed_loop and motion_armed and callable(getattr(ctrl, "apply_gui_targets_deg", None)):
                    self.status_text.set(
                        "Status: RUNNING — đổi Target Motor 0/1/2 (deg) sẽ cập nhật mục tiêu ngay."
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
                    self._set_entry(self.entry_err0, f"{ps0 - p0:.3f}")
                    self._set_entry(self.entry_err1, f"{ps1 - p1:.3f}")
                    self._set_entry(self.entry_err2, f"{ps2 - p2:.3f}")

                # ── Update plot ───────────────────────────────────────────
                if self.plotting and ctrl and connected:
                    data = ctrl.get_data()
                    if data:
                        # data tuple: (time, pos0, pos1, pos2, pos0_set, pos1_set, pos2_set)
                        times     = [d[0] for d in data]
                        pos0_vals = [d[1] for d in data]
                        pos1_vals = [d[2] for d in data]
                        pos2_vals = [d[3] for d in data]
                        set0_vals = [d[4] for d in data]
                        set1_vals = [d[5] for d in data]
                        set2_vals = [d[6] for d in data]
                        err0_vals = [d[4] - d[1] for d in data]
                        err1_vals = [d[5] - d[2] for d in data]
                        err2_vals = [d[6] - d[3] for d in data]

                        t0 = times[0] if self._last_t0 is None else self._last_t0
                        if self._last_t0 is None or (times[-1] - t0) > 30.0:
                            t0 = times[0]
                            self._last_t0 = t0
                        t_rel = [t - t0 for t in times]

                        self._line_pos0.set_data(t_rel, pos0_vals)
                        self._line_pos1.set_data(t_rel, pos1_vals)
                        self._line_pos2.set_data(t_rel, pos2_vals)
                        self._line_pos0_set.set_data(t_rel, set0_vals)
                        self._line_pos1_set.set_data(t_rel, set1_vals)
                        self._line_pos2_set.set_data(t_rel, set2_vals)
                        self._line_err0.set_data(t_rel, err0_vals)
                        self._line_err1.set_data(t_rel, err1_vals)
                        self._line_err2.set_data(t_rel, err2_vals)

                        self.ax_pos.relim(); self.ax_pos.autoscale_view()
                        self.ax_vel.relim(); self.ax_vel.autoscale_view()
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