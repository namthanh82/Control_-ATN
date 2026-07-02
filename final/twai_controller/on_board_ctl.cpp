// =============================================================================
//  on_board_ctl.cpp — implementation cho control on-board
// =============================================================================
#include "on_board_ctl.h"
#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"

// Joint zero offsets
constexpr float JOINT0_ZERO_DEG    = 0.0f;
constexpr float JOINT1_ZERO_DEG    = 0.0f;
constexpr float JOINT2_ZERO_DEG    = 0.0f;
constexpr int32_t JOINT0_ZERO_COUNT = 0;
constexpr int32_t JOINT1_ZERO_COUNT = 0;
constexpr int32_t JOINT2_ZERO_COUNT = 0;

CtrlState g_ctrl;

// Interrupt counters (defined in .ino) - kept for compatibility
extern volatile int32_t isr_count0;
extern volatile int32_t isr_count1;
extern volatile int32_t isr_count2;

extern int32_t readEncoderCount(const EncoderDef& enc);
extern float encoderCountToDeg(const EncoderDef& enc, int32_t count, float zero_deg, int32_t zero_count);

constexpr int8_t ENC_ID_0 = 0;
constexpr int8_t ENC_ID_1 = 1;
constexpr int8_t ENC_ID_2 = 2;

extern EncoderDef enc0;
extern EncoderDef enc1;
extern EncoderDef enc2;

extern volatile float joint_deg0;
extern volatile float joint_deg1;
extern volatile float joint_deg2;

extern float last_pos0, last_pos1, last_pos2;

extern void sendInputTorque(uint8_t node_id, float torque_nm);

static portMUX_TYPE s_ctrl_mux = portMUX_INITIALIZER_UNLOCKED;

// ============================================================================
//  CTC compute in main loop (called from .ino loop())
// ============================================================================
void onBoardCtlUpdate() {
    if (!g_ctrl.motion_active) {
        g_ctrl.tau_joint[0] = g_ctrl.tau_joint[1] = g_ctrl.tau_joint[2] = 0.0f;
        return;
    }

    // Update current position from encoders
    int32_t cnt0 = readEncoderCount(enc0);
    int32_t cnt1 = readEncoderCount(enc1);
    int32_t cnt2 = readEncoderCount(enc2);
    float q0 = encoderCountToDeg(enc0, cnt0, JOINT0_ZERO_DEG, JOINT0_ZERO_COUNT);
    float q1 = encoderCountToDeg(enc1, cnt1, JOINT1_ZERO_DEG, JOINT1_ZERO_COUNT);
    float q2 = encoderCountToDeg(enc2, cnt2, JOINT2_ZERO_DEG, JOINT2_ZERO_COUNT);

    // Compute velocity from position delta (LP filter)
    static float qdot_deg[3] = {0,0,0};
    static float q_prev[3] = {0,0,0};
    static uint32_t t_prev_ms = 0;
    uint32_t t_now_ms = millis();
    uint32_t dt_ms = t_now_ms - t_prev_ms;
    if (t_prev_ms == 0 || dt_ms > 100) {
        // First call or gap - reset
        qdot_deg[0] = qdot_deg[1] = qdot_deg[2] = 0;
        q_prev[0] = q0; q_prev[1] = q1; q_prev[2] = q2;
        t_prev_ms = t_now_ms;
    } else {
        float dt_s = dt_ms * 0.001f;
        float alpha = 0.3f;  // LP filter
        qdot_deg[0] = alpha * (q0 - q_prev[0]) / dt_s + (1.0f - alpha) * qdot_deg[0];
        qdot_deg[1] = alpha * (q1 - q_prev[1]) / dt_s + (1.0f - alpha) * qdot_deg[1];
        qdot_deg[2] = alpha * (q2 - q_prev[2]) / dt_s + (1.0f - alpha) * qdot_deg[2];
        q_prev[0] = q0; q_prev[1] = q1; q_prev[2] = q2;
        t_prev_ms = t_now_ms;
    }

    g_ctrl.q_deg[0] = q0;
    g_ctrl.q_deg[1] = q1;
    g_ctrl.q_deg[2] = q2;
    g_ctrl.qdot_deg[0] = qdot_deg[0];
    g_ctrl.qdot_deg[1] = qdot_deg[1];
    g_ctrl.qdot_deg[2] = qdot_deg[2];

    // Time since motion started (real wall-clock time)
    static uint32_t motion_start_ms = 0;
    if (g_ctrl.startup_t_s <= 0.0f) {
        motion_start_ms = t_now_ms;
        g_ctrl.startup_t_s = 0.001f;  // avoid division by zero
    }
    float t_prog = (t_now_ms - motion_start_ms) * 0.001f;

    float q_des[3], qd_des[3], qdd_des[3];
    bool done = true;
    for (int i=0;i<3;i++) {
        if (t_prog >= g_ctrl.traj[i].total_time) {
            q_des[i]   = g_ctrl.traj[i].end_p;
            qd_des[i]  = 0.0f;
            qdd_des[i] = 0.0f;
        } else {
            g_ctrl.traj[i].desired(t_prog, q_des[i], qd_des[i], qdd_des[i]);
            done = false;
        }
    }

    float home_offset[3];
    for (int i=0;i<3;i++) home_offset[i] = g_ctrl.ctc.model.model_home_deg[i];
    
    float q_rad[3], qd_rad[3], qdot_rad[3], qd_dot_rad[3], qdd_ddot_rad[3];
    for (int i=0;i<3;i++) {
        q_rad[i]        = (g_ctrl.q_deg[i] + home_offset[i])   * DEG2RAD;
        qd_rad[i]       = (q_des[i] + home_offset[i])           * DEG2RAD;
        
        // Sanitize velocity - clamp NaN/inf to 0 to avoid CTC compute failure
        float raw_vel = g_ctrl.qdot_deg[i];
        if (!isfinite(raw_vel) || fabsf(raw_vel) > 1000.0f) {
            raw_vel = 0.0f;
        }
        qdot_rad[i]     = raw_vel * DEG2RAD;
        qd_dot_rad[i]   = qd_des[i]                             * DEG2RAD;
        qdd_ddot_rad[i] = qdd_des[i]                            * DEG2RAD;
    }

    float tau_joint[3];
    g_ctrl.ctc.compute(q_rad, qdot_rad, qd_rad, qd_dot_rad, qdd_ddot_rad, tau_joint, t_prog);

    g_ctrl.tau_joint[0] = tau_joint[0];
    g_ctrl.tau_joint[1] = tau_joint[1];
    g_ctrl.tau_joint[2] = tau_joint[2];

    // Debug print every 500ms
    static uint32_t last_print_ms = 0;
    uint32_t now = millis();
    if (now - last_print_ms >= 500) {
        last_print_ms = now;
        Serial.printf("[CTC] tau=(%.4f,%.4f,%.4f) q=(%.2f,%.2f,%.2f) t=%.3f done=%d\n",
                     tau_joint[0], tau_joint[1], tau_joint[2],
                     q0, q1, q2, t_prog, done);
    }

    if (done) {
        g_ctrl.motion_active = false;
        g_ctrl.startup_t_s = 0.0f;  // reset for next motion
    }
}

// ============================================================================
// ============================================================================
//  Init
// ============================================================================
void onBoardCtlInit() {
    for (int i=0;i<3;i++) {
        g_ctrl.traj[i].max_jerk = 10.0f;
        g_ctrl.traj[i].max_acc  = 5.0f;
    }
    Serial.println("INFO: on-board CTC controller ready (main loop)");
}

void onBoardCtlSetTargetDeg(float q0, float q1, float q2) {
    portENTER_CRITICAL(&s_ctrl_mux);
    float cur[3] = {g_ctrl.q_deg[0], g_ctrl.q_deg[1], g_ctrl.q_deg[2]};
    float vmax   = g_ctrl.traj_max_vel_deg_s;
    portEXIT_CRITICAL(&s_ctrl_mux);

    g_ctrl.traj[0].compute(cur[0], q0, vmax);
    g_ctrl.traj[1].compute(cur[1], q1, vmax);
    g_ctrl.traj[2].compute(cur[2], q2, vmax);

    portENTER_CRITICAL(&s_ctrl_mux);
    g_ctrl.startup_t_s = 0.0f;  // signal to reset motion_start_ms in onBoardCtlUpdate
    g_ctrl.motion_active = true;
    portEXIT_CRITICAL(&s_ctrl_mux);

    Serial.printf("INFO: GOTO (%.2f, %.2f, %.2f) from (%.2f, %.2f, %.2f) vmax=%.1f\n",
                  q0, q1, q2, cur[0], cur[1], cur[2], vmax);
}

void onBoardCtlHoldCurrent() {
    portENTER_CRITICAL(&s_ctrl_mux);
    g_ctrl.motion_active = false;
    g_ctrl.startup_t_s = 0.0f;
    g_ctrl.tau_joint[0] = g_ctrl.tau_joint[1] = g_ctrl.tau_joint[2] = 0.0f;
    portEXIT_CRITICAL(&s_ctrl_mux);
    sendInputTorque(0, 0.0f);
    sendInputTorque(1, 0.0f);
    sendInputTorque(2, 0.0f);
    Serial.println("INFO: HOLD — torque set to 0");
}

void onBoardCtlSetKpKd(float kp0, float kp1, float kp2,
                       float kd0, float kd1, float kd2) {
    g_ctrl.ctc.gains.kp[0] = kp0;
    g_ctrl.ctc.gains.kp[1] = kp1;
    g_ctrl.ctc.gains.kp[2] = kp2;
    g_ctrl.ctc.gains.kd[0] = kd0;
    g_ctrl.ctc.gains.kd[1] = kd1;
    g_ctrl.ctc.gains.kd[2] = kd2;
    Serial.printf("INFO: Kp=(%.2f,%.2f,%.2f) Kd=(%.2f,%.2f,%.2f)\n",
                  kp0, kp1, kp2, kd0, kd1, kd2);
}

void onBoardCtlSetPrismatic(float hip_mm, float knee_mm) {
    hip_mm = constrain(hip_mm, PRISMATIC_MIN_MM, PRISMATIC_MAX_MM);
    knee_mm = constrain(knee_mm, PRISMATIC_MIN_MM, PRISMATIC_MAX_MM);
    
    portENTER_CRITICAL(&s_ctrl_mux);
    g_ctrl.prismatic_hip_mm = hip_mm;
    g_ctrl.prismatic_knee_mm = knee_mm;
    g_ctrl.ctc.model.hip_length = hip_mm / 1000.0f;
    g_ctrl.ctc.model.knee_length = knee_mm / 1000.0f;
    portEXIT_CRITICAL(&s_ctrl_mux);
    
    Serial.printf("INFO: Prismatic set - hip=%.1fmm l1=%.3fm, knee=%.1fmm l2=%.3fm\n",
                  hip_mm, hip_mm/1000.0f, knee_mm, knee_mm/1000.0f);
}

void onBoardCtlGetPrismatic(float& hip_mm, float& knee_mm) {
    portENTER_CRITICAL(&s_ctrl_mux);
    hip_mm = g_ctrl.prismatic_hip_mm;
    knee_mm = g_ctrl.prismatic_knee_mm;
    portEXIT_CRITICAL(&s_ctrl_mux);
}

float jointToMotorTorque(int axis, float tau_joint) {
    if (axis < 0 || axis > 2) return 0.0f;
    float gear = g_ctrl.gear_ratios[axis];
    float eff  = g_ctrl.motor_efficiency[axis];
    if (eff < 1e-6f) eff = 1.0f;
    return tau_joint / (gear * eff);
}

void onBoardCtlSetLockedAxes(bool lock0, bool lock1, bool lock2) {
    portENTER_CRITICAL(&s_ctrl_mux);
    g_ctrl.locked_axes[0] = lock0;
    g_ctrl.locked_axes[1] = lock1;
    g_ctrl.locked_axes[2] = lock2;
    portEXIT_CRITICAL(&s_ctrl_mux);
    Serial.printf("INFO: Locked axes set - %d,%d,%d\n", lock0, lock1, lock2);
}
