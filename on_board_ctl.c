/**
 * On-Board Controller Implementation
 * CTC Dynamics + Spline Trajectory
 */

#include "on_board_ctl.h"
#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"
#include "esp_log.h"

static const char* TAG = "CTRL";

// Joint zero offsets
#define JOINT0_ZERO_DEG 0.0f
#define JOINT1_ZERO_DEG 0.0f
#define JOINT2_ZERO_DEG 0.0f
#define JOINT0_ZERO_COUNT 0
#define JOINT1_ZERO_COUNT 0
#define JOINT2_ZERO_COUNT 0

static portMUX_TYPE s_ctrl_mux = portMUX_INITIALIZER_UNLOCKED;
static uint32_t s_motion_start_ms = 0;

// ============================================================================
// CTC Compute (called from main loop)
// ============================================================================
void onBoardCtlUpdate() {
    if (!g_ctrl.motion_active) {
        g_ctrl.tau_joint[0] = g_ctrl.tau_joint[1] = g_ctrl.tau_joint[2] = 0.0f;
        return;
    }

    uint32_t t_now_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
    
    // Time since motion started
    if (g_ctrl.startup_t_s <= 0.0f) {
        s_motion_start_ms = t_now_ms;
        g_ctrl.startup_t_s = 0.001f;
    }
    float t_prog = (t_now_ms - s_motion_start_ms) * 0.001f;

    // Get current state from encoder (these should be updated externally)
    extern float g_joint_pos_deg[3];
    extern float g_joint_vel_deg_s[3];

    // Trajectory desired state
    float q_des[3], qd_des[3], qdd_des[3];
    bool done = true;
    for (int i = 0; i < 3; i++) {
        if (t_prog >= g_ctrl.traj[i].total_time) {
            q_des[i] = g_ctrl.traj[i].end_p;
            qd_des[i] = 0.0f;
            qdd_des[i] = 0.0f;
        } else {
            traj_desired(&g_ctrl.traj[i], t_prog, &q_des[i], &qd_des[i], &qdd_des[i]);
            done = false;
        }
    }

    // DEBUG: Log trajectory state every 500ms
    static uint32_t s_last_debug_ms = 0;
    if (t_now_ms - s_last_debug_ms > 500) {
        s_last_debug_ms = t_now_ms;
        ESP_LOGI(TAG, "CTC t=%.3f traj_t=[%.2f,%.2f,%.2f] q_des=[%.2f,%.2f,%.2f] q_cur=[%.2f,%.2f,%.2f]",
            t_prog,
            g_ctrl.traj[0].total_time, g_ctrl.traj[1].total_time, g_ctrl.traj[2].total_time,
            q_des[0], q_des[1], q_des[2],
            g_joint_pos_deg[0], g_joint_pos_deg[1], g_joint_pos_deg[2]);
    }
    
    float q_rad[3], qd_rad[3], qdot_rad[3], qd_dot_rad[3], qdd_ddot_rad[3];
    for (int i = 0; i < 3; i++) {
        float home_offset = g_ctrl.ctc.model.model_home_deg[i];
        q_rad[i] = (g_joint_pos_deg[i] + home_offset) * DEG2RAD;
        qd_rad[i] = q_des[i] * DEG2RAD;  // q_des đã ở joint space, không cần home_offset
        
        float raw_vel = g_joint_vel_deg_s[i];
        if (!isfinite(raw_vel) || fabsf(raw_vel) > 1000.0f) raw_vel = 0.0f;
        qdot_rad[i] = raw_vel * DEG2RAD;
        qd_dot_rad[i] = qd_des[i] * DEG2RAD;
        qdd_ddot_rad[i] = qdd_des[i] * DEG2RAD;
    }

    float tau_joint[3];
    ctc_compute(&g_ctrl.ctc, q_rad, qdot_rad, qd_rad, qd_dot_rad, qdd_ddot_rad, tau_joint, t_prog);

    // DEBUG: Log torque
    static uint32_t s_last_tau_ms = 0;
    if (t_now_ms - s_last_tau_ms > 500) {
        s_last_tau_ms = t_now_ms;
        ESP_LOGI(TAG, "CTC tau_joint=[%.3f,%.3f,%.3f] tau_motor=[%.4f,%.4f,%.4f]",
            tau_joint[0], tau_joint[1], tau_joint[2],
            tau_joint[0] / (g_ctrl.gear_ratios[0] * g_ctrl.motor_efficiency[0]),
            tau_joint[1] / (g_ctrl.gear_ratios[1] * g_ctrl.motor_efficiency[1]),
            tau_joint[2] / (g_ctrl.gear_ratios[2] * g_ctrl.motor_efficiency[2]));
    }

    g_ctrl.tau_joint[0] = tau_joint[0];
    g_ctrl.tau_joint[1] = tau_joint[1];
    g_ctrl.tau_joint[2] = tau_joint[2];

    if (done) {
        g_ctrl.motion_active = false;
        g_ctrl.startup_t_s = 0.0f;
    }
}

// ============================================================================
// Init
// ============================================================================
void onBoardCtlInit() {
    ctc_model_init(&g_ctrl.ctc.model);
    gains_init(&g_ctrl.ctc.gains);
    
    g_ctrl.traj_max_vel_deg_s = 25.0f;
    g_ctrl.startup_t_s = 0.0f;
    g_ctrl.motion_active = false;
    g_ctrl.tau_joint[0] = g_ctrl.tau_joint[1] = g_ctrl.tau_joint[2] = 0.0f;
    g_ctrl.locked_axes[0] = g_ctrl.locked_axes[1] = g_ctrl.locked_axes[2] = false;
    g_ctrl.gear_ratios[0] = 100.0f;
    g_ctrl.gear_ratios[1] = 50.0f;
    g_ctrl.gear_ratios[2] = 50.0f;
    g_ctrl.motor_efficiency[0] = 1.0f;
    g_ctrl.motor_efficiency[1] = 1.0f;
    g_ctrl.motor_efficiency[2] = 1.0f;
    
    for (int i = 0; i < 3; i++) {
        g_ctrl.traj[i].max_jerk = 10.0f;
        g_ctrl.traj[i].max_acc = 5.0f;
    }
    
    ESP_LOGI(TAG, "CTC controller initialized");
}

void onBoardCtlSetTargetDeg(float q0, float q1, float q2) {
    // Read current time BEFORE critical section (safe)
    uint32_t t_now_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;

    portENTER_CRITICAL(&s_ctrl_mux);
    // Use g_joint_pos_deg (updated from encoder) instead of g_ctrl.q_deg (never updated)
    extern float g_joint_pos_deg[3];
    float cur[3] = {g_joint_pos_deg[0], g_joint_pos_deg[1], g_joint_pos_deg[2]};
    float vmax = g_ctrl.traj_max_vel_deg_s;
    portEXIT_CRITICAL(&s_ctrl_mux);

    traj_compute(&g_ctrl.traj[0], cur[0], q0, vmax);
    traj_compute(&g_ctrl.traj[1], cur[1], q1, vmax);
    traj_compute(&g_ctrl.traj[2], cur[2], q2, vmax);

    // Reset motion timer so trajectory starts NOW
    s_motion_start_ms = t_now_ms;
    g_ctrl.startup_t_s = 0.001f;  // Non-zero to skip re-init in onBoardCtlUpdate

    portENTER_CRITICAL(&s_ctrl_mux);
    g_ctrl.motion_active = true;
    portEXIT_CRITICAL(&s_ctrl_mux);

    ESP_LOGI(TAG, "GOTO (%.2f, %.2f, %.2f) from (%.2f, %.2f, %.2f)", q0, q1, q2, cur[0], cur[1], cur[2]);
}

void onBoardCtlHoldCurrent() {
    portENTER_CRITICAL(&s_ctrl_mux);
    g_ctrl.motion_active = false;
    g_ctrl.startup_t_s = 0.0f;
    g_ctrl.tau_joint[0] = g_ctrl.tau_joint[1] = g_ctrl.tau_joint[2] = 0.0f;
    portEXIT_CRITICAL(&s_ctrl_mux);
    
    ESP_LOGI(TAG, "HOLD - torque set to 0");
}

void onBoardCtlSetLockedAxes(bool lock0, bool lock1, bool lock2) {
    portENTER_CRITICAL(&s_ctrl_mux);
    g_ctrl.locked_axes[0] = lock0;
    g_ctrl.locked_axes[1] = lock1;
    g_ctrl.locked_axes[2] = lock2;
    portEXIT_CRITICAL(&s_ctrl_mux);
    
    ESP_LOGI(TAG, "Locked axes: %d,%d,%d", lock0, lock1, lock2);
}

float jointToMotorTorque(int axis, float tau_joint) {
    if (axis < 0 || axis > 2) return 0.0f;
    float gear = g_ctrl.gear_ratios[axis];
    float eff = g_ctrl.motor_efficiency[axis];
    if (eff < 1e-6f) eff = 1.0f;
    return tau_joint / (gear * eff);
}
