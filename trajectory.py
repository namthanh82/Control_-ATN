from abc import ABC, abstractmethod
import math


class TrajectoryBase(ABC):
    def __init__(self):
        # Khởi tạo các biến lưu trữ tham số đã tính toán
        self.start_p = -90.0
        self.end_p = -90.0
        self.maxVel = math.inf
        self.direction = 1.0
        self.total_time = math.inf

    @abstractmethod
    def param_calc(self, start_p, end_p, max_v):
        """
        Method này chạy NẶNG, chứa logic phức tạp (căn bậc 2, giải phương trình...).
        Chỉ gọi 1 lần khi có lệnh Move mới.
        """
        pass

    @abstractmethod
    def desired_state(self, t):
        """
        Method này chạy NHẸ (chỉ cộng trừ nhân chia đơn giản).
        Gọi liên tục trong vòng lặp timer (Real-time).
        Output: pos, vel, acc
        """
        pass

    def reset(self):
        self.start_p = -90.0
        self.end_p = -90.0
        self.total_time = math.inf
        self.direction = 1.0


class TrapezoidalTrajectory(TrajectoryBase):
    def __init__(self):
        super().__init__()
        # Các biến riêng của hình thang
        self.j_peak = 4  # deg/s^3
        self.accel = 10.0  # deg/s^2 - gia tốc cố định
        self.v_peak = 0.0  # deg/s
        self.t_acc = 0.0  # Thời gian tăng tốc
        self.t_dec = 0.0  # Thời điểm bắt đầu giảm tốc

    def param_calc(self, start_p, end_p, max_v):
        self.start_p = start_p
        self.end_p = end_p
        distance = end_p - start_p
        self.direction = 1.0 if distance >= 0 else -1.0
        abs_dist = abs(distance)

        if abs_dist < 0.005:
            self.v_peak = 0.0
            self.t_acc = 0.0
            self.total_time = 0.0
            self.t_dec = 0.0
            return

        # Tính thời gian tăng tốc
        self.t_acc = max_v / self.accel
        d_acc = 0.5 * self.accel * self.t_acc ** 2

        if abs_dist <= 2 * d_acc:
            # Không có giai đoạn tốc độ đều
            self.v_peak = math.sqrt(abs_dist * self.accel)
            self.t_acc = self.v_peak / self.accel
            self.total_time = 2 * self.t_acc
            self.t_dec = self.t_acc
        else:
            # Có giai đoạn tốc độ đều
            self.v_peak = max_v
            d_const = abs_dist - 2 * d_acc
            t_const = d_const / max_v
            self.total_time = 2 * self.t_acc + t_const
            self.t_dec = self.t_acc + t_const

    def desired_state(self, t):
        # Xử lý ngoài phạm vi

        if t <= 0: return self.start_p, 0.0, 0.0
        if t >= self.total_time or self.total_time == math.inf: return self.end_p, 0.0, 0.0

        pos = 0.0
        vel = 0.0
        acc = 0.0

        if t < self.t_acc:  # Giai đoạn 1: Tăng tốc
            pos = 0.5 * self.accel * t * t
            vel = self.accel * t
            acc = self.accel

        elif t < self.t_dec:  # Giai đoạn 2: Tốc độ đều
            dt = t - self.t_acc
            pos = 0.5 * self.accel * self.t_acc ** 2 + self.v_peak * dt
            vel = self.v_peak
            acc = 0.0

        else:  # Giai đoạn 3: Giảm tốc
            t_rem = self.total_time - t
            dist_rem = 0.5 * self.accel * t_rem * t_rem
            pos = abs(self.end_p - self.start_p) - dist_rem
            vel = self.accel * t_rem
            acc = -self.accel

        # Kết hợp hướng
        final_p = self.start_p + pos * self.direction
        final_v = vel * self.direction
        final_a = acc * self.direction

        return final_p, final_v, final_a


class CubicTrajectory(TrajectoryBase):
    def __init__(self):
        super().__init__()
        # Hệ số phương trình: q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        self.a0 = 0
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0

    def param_calc(self, start_p, end_p, max_v):

        self.start_p = start_p
        self.end_p = end_p
        dist = end_p - start_p
        abs_dist = abs(dist)

        # Tính thời gian dựa trên max_v, giả sử thời gian = abs_dist / max_v * 2 để smooth và không vượt quá max_v quá nhiều
        self.total_time = abs_dist / max_v * 2.0 if max_v > 0 else 1.0

        T = self.total_time
        self.a0 = start_p
        self.a1 = 0
        self.a2 = 3 * dist / (T ** 2)
        self.a3 = -2 * dist / (T ** 3)

    def desired_state(self, t):
        if t <= 0:
            if getattr(self, "debug_trace", False):
                print(f"[TRAJ][TRACE] Spline desired_state t={t:.6f} -> start={self.start_p:.6f}")
            return self.start_p, 0.0, 0.0
        if t >= self.total_time or self.total_time == math.inf:
            if getattr(self, "debug_trace", False):
                print(f"[TRAJ][TRACE] Spline desired_state t={t:.6f} -> end={self.end_p:.6f}")
            return self.end_p, 0.0, 0.0

        t2 = t * t
        t3 = t2 * t

        pos = self.a0 + self.a1 * t + self.a2 * t2 + self.a3 * t3
        vel = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t2
        acc = 2 * self.a2 + 6 * self.a3 * t

        return pos, vel, acc


class QuinticTrajectory(TrajectoryBase):
    def __init__(self):
        super().__init__()
        # Hệ số phương trình bậc 5: q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        self.a0 = 0
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = 0

    def param_calc(self, start_p, end_p, max_v):
        self.start_p = start_p
        self.end_p = end_p
        dist = end_p - start_p

        if getattr(self, "debug_trace", False):
            print(f"[TRAJ][TRACE] Quintic param_calc start={start_p:.6f} end={end_p:.6f} dist={dist:.6f} max_v={max_v:.6f}")

        abs_dist = abs(dist)
        if abs_dist == 0.0:
            self.total_time = 0.0
            self.a0 = start_p
            self.a1 = 0.0
            self.a2 = 0.0
            self.a3 = 0.0
            self.a4 = 0.0
            self.a5 = 0.0
            if getattr(self, "debug_trace", False):
                print(f"[TRAJ][TRACE] Quintic param_calc stationary start=end={start_p:.6f}")
            return

        # Tính thời gian dựa trên max_v, giả sử thời gian = abs(dist) / max_v * 2.5 để smooth
        self.total_time = abs_dist / max_v * 2.5 if max_v > 0 else 1.0

        T = self.total_time
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        self.a0 = start_p
        self.a1 = 0
        self.a2 = 0
        self.a3 = 10 * dist / T3
        self.a4 = -15 * dist / T4
        self.a5 = 6 * dist / T5

    def desired_state(self, t):
        if t <= 0:
            if getattr(self, "debug_trace", False):
                print(f"[TRAJ][TRACE] Quintic desired_state t={t:.6f} -> start={self.start_p:.6f}")
            return self.start_p, 0.0, 0.0
        if t >= self.total_time or self.total_time == math.inf:
            if getattr(self, "debug_trace", False):
                print(f"[TRAJ][TRACE] Quintic desired_state t={t:.6f} -> end={self.end_p:.6f}")
            return self.end_p, 0.0, 0.0

        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        pos = self.a0 + self.a1 * t + self.a2 * t2 + self.a3 * t3 + self.a4 * t4 + self.a5 * t5
        vel = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t2 + 4 * self.a4 * t3 + 5 * self.a5 * t4
        acc = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t2 + 20 * self.a5 * t3

        if getattr(self, "debug_trace", False):
            print(f"[TRAJ][TRACE] Quintic desired_state t={t:.6f} pos={pos:.6f} vel={vel:.6f} acc={acc:.6f} start={self.start_p:.6f} end={self.end_p:.6f}")

        return pos, vel, acc


class SplineTrajectory(TrajectoryBase):
    def __init__(self):
        super().__init__()
        # Các tham số cho quỹ đạo splines 7 đoạn với giới hạn jerk
        self.max_jerk = 40 # deg/s^3
        self.max_acc = 20  # deg/s^2
        self.total_time = 0.0
        # Các thời gian cho từng phase
        self.t1 = 0.0
        self.t2 = 0.0
        self.t3 = 0.0
        self.t4 = 0.0
        self.t5 = 0.0
        self.t6 = 0.0
        self.t7 = 0.0
        # Các khoảng cách tích lũy (tùy chọn, để debug)
        self.d1 = 0.0
        self.d2 = 0.0
        self.d3 = 0.0
        self.d4 = 0.0
        self.d5 = 0.0
        self.d6 = 0.0
        self.d7 = 0.0
        # Các biến vận tốc tại các giai đoạn
        self.v1 = 0.0
        self.v2 = 0.0
        self.v3 = 0.0
        self.v4 = 0.0
        self.v5 = 0.0
        self.v6 = 0.0
        self.v7 = 0.0
        # Đỉnh gia tốc thực tế
        self.a_pk = 0.0
    def param_calc(self, start_p, end_p, max_v):
        self.start_p = start_p
        self.end_p = end_p
        dist = end_p - start_p
        abs_dist = abs(dist)
        self.direction = 1.0 if dist >= 0 else -1.0
        if getattr(self, "debug_trace", False):
            print(f"[TRAJ][TRACE] Spline param_calc start={start_p:.6f} end={end_p:.6f} dist={dist:.6f} dir={self.direction:+.1f} max_v={max_v:.6f}")
        if abs_dist == 0.0:
            self.total_time = 0.0
            if getattr(self, "debug_trace", False):
                print(f"[TRAJ][TRACE] Spline param_calc stationary start=end={start_p:.6f}")
            return
        # Xác định khả năng chạm đỉnh gia tốc
        v_jerk_phase = (self.max_acc ** 2) / self.max_jerk
        if max_v < v_jerk_phase:
            a_pk = math.sqrt(max_v * self.max_jerk)
            t_j = a_pk / self.max_jerk
            t_a = 0.0
            d_req = max_v * (2 * t_j)
        else:
            a_pk = self.max_acc
            t_j = a_pk / self.max_jerk
            t_a = (max_v - v_jerk_phase) / a_pk
            # Khoảng cách cần thiết để vừa tăng tốc lên max_v rồi phanh lại
            d_req = max_v * (2 * t_j + t_a)

        if abs_dist >= d_req:
            t_const = (abs_dist - d_req) / max_v
            v_pk = max_v
        else:
            t_const = 0.0
            d_acc_limit = 2 * (self.max_acc ** 3) / (self.max_jerk ** 2)
            if abs_dist < d_acc_limit:
                a_pk = math.pow(abs_dist * (self.max_jerk ** 2) / 2.0, 1.0 / 3.0)
                t_j = a_pk / self.max_jerk
                t_a = 0.0
                v_pk = a_pk * t_j
            else:
                c = 2 * (t_j ** 2) - (abs_dist / a_pk)
                delta = 9 * (t_j ** 2) - 4 * c
                t_a = (-3 * t_j + math.sqrt(delta)) / 2.0
                v_pk = a_pk * t_j + a_pk * t_a

        self.t1 = t_j
        self.t2 = t_a
        self.t3 = t_j
        self.t4 = t_const
        self.t5 = t_j
        self.t6 = t_a
        self.t7 = t_j
        self.total_time = self.t1 + self.t2 + self.t3 + self.t4 + self.t5 + self.t6 + self.t7
        self.a_pk = a_pk
        j = self.max_jerk
        a = self.a_pk

        # Tính khoảng cách tích lũy tại cuối mỗi phase
        self.d1 = (1 / 6.0) * j * self.t1 ** 3
        self.v1 = (1 / 2.0) * j * self.t1 ** 2
        self.d2 = self.d1 + self.v1 * self.t2 + (1 / 2.0) * a * self.t2 ** 2
        self.v2 = self.v1 + a * self.t2
        self.v3 = self.v2 + a * self.t3 - (1 / 2.0) * j * self.t3 ** 2
        self.d3 = self.d2 + self.v2 * self.t3 + (1 / 2.0) * a * self.t3 ** 2 - (1 / 6.0) * j * self.t3 ** 3
        self.v4 = self.v3
        self.d4 = self.d3 + self.v3 * self.t4
        self.v5 = self.v4 - (1 / 2.0) * j * self.t5 ** 2
        self.d5 = self.d4 + self.v4 * self.t5 - (1 / 6.0) * j * self.t5 ** 3
        self.v6 = self.v5 - a * self.t6
        self.d6 = self.d5 + self.v5 * self.t6 - (1 / 2.0) * a * self.t6 ** 2

    def desired_state(self, t):
        if t <= 0: return self.start_p, 0.0, 0.0
        if t >= self.total_time or self.total_time == math.inf: return self.end_p, 0.0, 0.0

        j = self.max_jerk
        a = self.a_pk

        if t < self.t1:
            dt = t
            acc = j * dt
            vel = 0.5 * j * dt ** 2
            pos = (1 / 6.0) * j * dt ** 3

        elif t < self.t1 + self.t2:
            dt = t - self.t1
            acc = a
            vel = self.v1 + a * dt
            pos = self.d1 + self.v1 * dt + 0.5 * a * dt ** 2

        elif t < self.t1 + self.t2 + self.t3:
            dt = t - (self.t1 + self.t2)
            acc = a - j * dt
            vel = self.v2 + a * dt - 0.5 * j * dt ** 2
            pos = self.d2 + self.v2 * dt + 0.5 * a * dt ** 2 - (1 / 6.0) * j * dt ** 3

        elif t < self.t1 + self.t2 + self.t3 + self.t4:
            dt = t - (self.t1 + self.t2 + self.t3)
            acc = 0.0
            vel = self.v3
            pos = self.d3 + self.v3 * dt

        elif t < self.t1 + self.t2 + self.t3 + self.t4 + self.t5:
            dt = t - (self.t1 + self.t2 + self.t3 + self.t4)
            acc = -j * dt
            vel = self.v4 - 0.5 * j * dt ** 2
            pos = self.d4 + self.v4 * dt - (1 / 6.0) * j * dt ** 3

        elif t < self.t1 + self.t2 + self.t3 + self.t4 + self.t5 + self.t6:
            dt = t - (self.t1 + self.t2 + self.t3 + self.t4 + self.t5)
            acc = -a
            vel = self.v5 - a * dt
            pos = self.d5 + self.v5 * dt - 0.5 * a * dt ** 2

        else:
            dt = t - (self.t1 + self.t2 + self.t3 + self.t4 + self.t5 + self.t6)
            acc = -a + j * dt
            vel = self.v6 - a * dt + 0.5 * j * dt ** 2
            pos = self.d6 + self.v6 * dt - 0.5 * a * dt ** 2 + (1 / 6.0) * j * dt ** 3

        pos = self.start_p + pos * self.direction
        vel = vel * self.direction
        acc = acc * self.direction

        return pos, vel, acc