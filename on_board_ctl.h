/**
 * On-Board Controller for 3-DOF Robot Arm
 * Full CTC Control with Spline Trajectory Generation
 * Based on namthanh82/Control_-ATN
 */

#ifndef ON_BOARD_CTL_H
#define ON_BOARD_CTL_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================
#define DEG2RAD 0.017453292519943295f
#define RAD2DEG 57.29577951308232f
#define GRAVITY 9.81f
#define TORQUE_MAX_NM 100.0f
#define CTRL_HZ 1000
#define FB_HZ 100

#define PRISMATIC_MIN_MM 350.0f
#define PRISMATIC_MAX_MM 450.0f

// ============================================================================
// Encoder Definition
// ============================================================================
typedef struct {
    int pinA;
    int pinB;
    int pinZ;
    int8_t id;
    int8_t direction;
    int32_t counts_per_rev;
} EncoderDef;

// ============================================================================
// CTC Model Parameters (3-DOF Robot Arm)
// ============================================================================
typedef struct {
    // Masses
    float hip_mass1;
    float hip_mass2;
    float knee_mass1;
    float knee_mass2;
    float ankle_mass;
    
    // COM positions
    float hip_x1, hip_y1;
    float hip_x2_base, hip_y2;
    float knee_x1, knee_y1;
    float knee_x2_base, knee_y2;
    float ankle_x, ankle_y;
    
    // Inertias
    float hip_inertia;
    float knee_inertia;
    float ankle_inertia;
    
    // Link lengths
    float hip_length;
    float knee_length;
    float ankle_length;
    
    // Motor parameters
    float hip_gear_ratio;
    float knee_gear_ratio;
    float ankle_gear_ratio;
    
    // Friction
    float coulomb_friction[3];
    float viscous_friction[3];
    
    // Torque scale
    float torque_scale;
    
    // Home position offset
    float model_home_deg[3];
    
    // Computed COM
    float hip_chung_x, hip_chung_y;
    float knee_chung_x, knee_chung_y;
    
    // Dynamic COM (adjustable)
    float delta_x1;
    float delta_x2;
    
    // Gravity constant
    float gravity;
} CTC3Model;

static void ctc_model_init(CTC3Model* m) {
    m->hip_mass1 = 2.928f;
    m->hip_mass2 = 3.277f;
    m->knee_mass1 = 2.898f;
    m->knee_mass2 = 3.201f;
    m->ankle_mass = 0.896f;
    
    m->hip_x1 = 0.128f; m->hip_y1 = 0.02988f;
    m->hip_x2_base = 0.28705f; m->hip_y2 = -0.00314f;
    m->knee_x1 = 0.12843f; m->knee_y1 = 0.03047f;
    m->knee_x2_base = 0.28561f; m->knee_y2 = -0.0032f;
    m->ankle_x = 0.05723f; m->ankle_y = 0.0f;
    
    m->hip_inertia = 0.064f;
    m->knee_inertia = 0.058f;
    m->ankle_inertia = 0.00274688527f;
    
    m->hip_length = 0.35f;
    m->knee_length = 0.35f;
    m->ankle_length = 0.07f;
    
    m->hip_gear_ratio = 100.0f;
    m->knee_gear_ratio = 50.0f;
    m->ankle_gear_ratio = 50.0f;
    
    m->coulomb_friction[0] = 0.05f; m->coulomb_friction[1] = 0.05f; m->coulomb_friction[2] = 0.05f;
    m->viscous_friction[0] = 0.00276f; m->viscous_friction[1] = 0.00276f; m->viscous_friction[2] = 0.00276f;
    
    m->torque_scale = 1.0f;
    m->model_home_deg[0] = 0.0f; m->model_home_deg[1] = -90.0f; m->model_home_deg[2] = 0.0f;
    m->delta_x1 = 0.0f; m->delta_x2 = 0.0f;
    m->gravity = GRAVITY;
}

static void ctc_model_update_com(CTC3Model* m) {
    float hip_x2 = m->hip_x2_base + m->delta_x1;
    float knee_x2 = m->knee_x2_base + m->delta_x2;
    
    float m01 = m->hip_mass1, m02 = m->hip_mass2;
    float hip_total_mass = m01 + m02;
    m->hip_chung_x = (m01 * m->hip_x1 + m02 * hip_x2) / hip_total_mass;
    m->hip_chung_y = (m01 * m->hip_y1 + m02 * m->hip_y2) / hip_total_mass;
    
    float m11 = m->knee_mass1, m12 = m->knee_mass2;
    float knee_total_mass = m11 + m12;
    m->knee_chung_x = (m11 * m->knee_x1 + m12 * knee_x2) / knee_total_mass;
    m->knee_chung_y = (m11 * m->knee_y1 + m12 * m->knee_y2) / knee_total_mass;
}

// ============================================================================
// PD Gains
// ============================================================================
typedef struct {
    float kp[3];
    float kd[3];
} Gains;

static void gains_init(Gains* g) {
    g->kp[0] = 3.0f; g->kp[1] = 6.0f; g->kp[2] = 3.0f;
    g->kd[0] = 1.0f; g->kd[1] = 0.5f; g->kd[2] = 1.0f;
}

// ============================================================================
// Spline Trajectory (7-segment trapezoidal with jerk)
// ============================================================================
typedef struct {
    float max_jerk;
    float max_acc;
    float max_vel;
    float start_p;
    float end_p;
    float direction;
    float total_time;
    float t1, t2, t3, t4, t5, t6, t7;
    float d1, d2, d3, d4, d5, d6;
    float v1, v2, v3, v4, v5, v6;
    float a_pk;
} SplineTrajectory;

static void traj_compute(SplineTrajectory* t, float start, float end, float max_v) {
    t->start_p = start;
    t->end_p = end;
    t->max_vel = max_v;
    float dist = end - start;
    float abs_dist = fabsf(dist);
    t->direction = (dist >= 0.0f) ? 1.0f : -1.0f;
    
    if (abs_dist < 1e-6f) {
        t->total_time = 0.0f;
        t->t1 = t->t2 = t->t3 = t->t4 = t->t5 = t->t6 = t->t7 = 0.0f;
        t->d1 = t->d2 = t->d3 = t->d4 = t->d5 = t->d6 = 0.0f;
        t->v1 = t->v2 = t->v3 = t->v4 = t->v5 = t->v6 = 0.0f;
        t->a_pk = 0.0f;
        return;
    }
    
    float v_jerk_phase = (t->max_acc * t->max_acc) / t->max_jerk;
    float t_j = 0.0f, t_a = 0.0f, t_const = 0.0f, v_pk = 0.0f, a_pk_local = 0.0f;
    
    if (max_v < v_jerk_phase) {
        a_pk_local = sqrtf(max_v * t->max_jerk);
        t_j = a_pk_local / t->max_jerk;
        float d_req = max_v * (2.0f * t_j);
        if (abs_dist >= d_req) {
            t_const = (abs_dist - d_req) / max_v;
            v_pk = max_v;
        } else {
            t_const = 0.0f;
            v_pk = a_pk_local * t_j;
        }
    } else {
        a_pk_local = t->max_acc;
        t_j = a_pk_local / t->max_jerk;
        t_a = (max_v - v_jerk_phase) / a_pk_local;
        float d_req = max_v * (2.0f * t_j + t_a);
        if (abs_dist >= d_req) {
            t_const = (abs_dist - d_req) / max_v;
            v_pk = max_v;
        } else {
            t_const = 0.0f;
            float d_acc_limit = 2.0f * powf(a_pk_local, 3.0f) / (t->max_jerk * t->max_jerk);
            if (abs_dist < d_acc_limit) {
                a_pk_local = cbrtf(abs_dist * t->max_jerk * t->max_jerk / 2.0f);
                t_j = a_pk_local / t->max_jerk;
                t_a = 0.0f;
                v_pk = a_pk_local * t_j;
            } else {
                float c = 2.0f * t_j * t_j - abs_dist / a_pk_local;
                float delta = 9.0f * t_j * t_j - 4.0f * c;
                if (delta < 0) delta = 0.0f;
                t_a = (-3.0f * t_j + sqrtf(delta)) / 2.0f;
                v_pk = a_pk_local * t_j + a_pk_local * t_a;
            }
        }
    }
    
    t->t1 = t->t3 = t->t5 = t->t7 = t_j;
    t->t2 = t->t6 = t_a;
    t->t4 = t_const;
    t->total_time = t->t1 + t->t2 + t->t3 + t->t4 + t->t5 + t->t6 + t->t7;
    t->a_pk = a_pk_local;
    
    float j = t->max_jerk;
    float a = a_pk_local;
    t->d1 = (1.0f/6.0f) * j * t->t1 * t->t1 * t->t1;
    t->v1 = 0.5f * j * t->t1 * t->t1;
    t->d2 = t->d1 + t->v1 * t->t2 + 0.5f * a * t->t2 * t->t2;
    t->v2 = t->v1 + a * t->t2;
    t->v3 = t->v2 + a * t->t3 - 0.5f * j * t->t3 * t->t3;
    t->d3 = t->d2 + t->v2 * t->t3 + 0.5f * a * t->t3 * t->t3 - (1.0f/6.0f) * j * t->t3 * t->t3 * t->t3;
    t->v4 = t->v3;
    t->d4 = t->d3 + t->v3 * t->t4;
    t->v5 = t->v4 - 0.5f * j * t->t5 * t->t5;
    t->d5 = t->d4 + t->v4 * t->t5 - (1.0f/6.0f) * j * t->t5 * t->t5 * t->t5;
    t->v6 = t->v5 - a * t->t6;
    t->d6 = t->d5 + t->v5 * t->t6 - 0.5f * a * t->t6 * t->t6;
}

static void traj_desired(const SplineTrajectory* t, float tt, float* pos, float* vel, float* acc) {
    if (tt <= 0.0f) { *pos = t->start_p; *vel = 0.0f; *acc = 0.0f; return; }
    if (tt >= t->total_time || t->total_time <= 0.0f) { *pos = t->end_p; *vel = 0.0f; *acc = 0.0f; return; }
    
    float j = t->max_jerk;
    float a = t->a_pk;
    float p = 0.0f, v = 0.0f, ac = 0.0f;
    
    if (tt < t->t1) {
        float dt = tt;
        ac = j * dt; v = 0.5f * j * dt * dt; p = (1.0f/6.0f) * j * dt * dt * dt;
    } else if (tt < t->t1 + t->t2) {
        float dt = tt - t->t1;
        ac = a; v = t->v1 + a * dt; p = t->d1 + t->v1 * dt + 0.5f * a * dt * dt;
    } else if (tt < t->t1 + t->t2 + t->t3) {
        float dt = tt - (t->t1 + t->t2);
        ac = a - j * dt; v = t->v2 + a * dt - 0.5f * j * dt * dt;
        p = t->d2 + t->v2 * dt + 0.5f * a * dt * dt - (1.0f/6.0f) * j * dt * dt * dt;
    } else if (tt < t->t1 + t->t2 + t->t3 + t->t4) {
        float dt = tt - (t->t1 + t->t2 + t->t3);
        ac = 0.0f; v = t->v3; p = t->d3 + t->v3 * dt;
    } else if (tt < t->t1 + t->t2 + t->t3 + t->t4 + t->t5) {
        float dt = tt - (t->t1 + t->t2 + t->t3 + t->t4);
        ac = -j * dt; v = t->v4 - 0.5f * j * dt * dt;
        p = t->d4 + t->v4 * dt - (1.0f/6.0f) * j * dt * dt * dt;
    } else if (tt < t->t1 + t->t2 + t->t3 + t->t4 + t->t5 + t->t6) {
        float dt = tt - (t->t1 + t->t2 + t->t3 + t->t4 + t->t5);
        ac = -a; v = t->v5 - a * dt; p = t->d5 + t->v5 * dt - 0.5f * a * dt * dt;
    } else {
        float dt = tt - (t->t1 + t->t2 + t->t3 + t->t4 + t->t5 + t->t6);
        ac = -a + j * dt; v = t->v6 - a * dt + 0.5f * j * dt * dt;
        p = t->d6 + t->v6 * dt - 0.5f * a * dt * dt + (1.0f/6.0f) * j * dt * dt * dt;
    }
    
    *pos = t->start_p + p * t->direction;
    *vel = v * t->direction;
    *acc = ac * t->direction;
}

// ============================================================================
// CTC 3-DOF Dynamics Computation
// ============================================================================
typedef struct {
    CTC3Model model;
    Gains gains;
} CTC3DOF;

static void ctc_compute(const CTC3DOF* ctc, 
                       const float q[3], const float q_dot[3],
                       const float qd[3], const float qd_dot[3], const float qd_ddot[3],
                       float tau_out[3], float startup_t) {
    const CTC3Model* m = &ctc->model;
    float g_const = m->gravity;
    
    ctc_model_update_com((CTC3Model*)m);
    
    float l1 = m->hip_length;
    float l2 = m->knee_length;
    
    float m0 = m->hip_mass1 + m->hip_mass2;
    float m1 = m->knee_mass1 + m->knee_mass2;
    float m2 = m->ankle_mass;
    float xc1 = m->hip_chung_x;
    float yc1 = m->hip_chung_y;
    float xc2 = m->knee_chung_x;
    float yc2 = m->knee_chung_y;
    float xc3 = m->ankle_x;
    float yc3 = m->ankle_y;
    
    float c1 = cosf(q[0]), s1 = sinf(q[0]);
    float c12 = cosf(q[0]+q[1]), s12 = sinf(q[0]+q[1]);
    float c123 = cosf(q[0]+q[1]+q[2]), s123 = sinf(q[0]+q[1]+q[2]);
    
    // Gravity terms
    float G0 = g_const * (m1*(xc2*c12 + l1*c1 - yc2*s12) 
            + m2*(l2*c12 + l1*c1 + xc3*c123 - yc3*s123) 
            + m0*(xc1*c1 - yc1*s1));
    float G1 = g_const * (m1*(xc2*c12 - yc2*s12) 
            + m2*(l2*c12 + xc3*c123 - yc3*s123));
    float G2 = g_const * (m2*xc3*c123 - m2*yc3*s123);
    
    // Startup blend
    const float startup_duration = 0.8f;
    float w = (startup_t / startup_duration);
    if (w < 0.0f) w = 0.0f;
    if (w > 1.0f) w = 1.0f;
    float G_blend[3] = {G0 * w, G1 * w, G2 * w};
    
    // Mass matrix elements
    float c2 = cosf(q[1]), s2 = sinf(q[1]);
    float c3 = cosf(q[2]), s3 = sinf(q[2]);
    float c23 = cosf(q[1]+q[2]), s23 = sinf(q[1]+q[2]);
    
    float Izz1 = m->hip_inertia, Izz2 = m->knee_inertia, Izz3 = m->ankle_inertia;
    
    float M00 = Izz1+Izz2+Izz3 + l1*l1*m1+l1*l1*m2+m0*(xc1*xc1+yc1*yc1)
              + m1*(xc2*xc2+yc2*yc2) + m2*(xc3*xc3+yc3*yc3)
              + 2.0f*(l1*m2*xc3*c23 + l1*l2*m2*c2 - l1*m2*yc3*s23 + l1*m1*xc2*c2 - l1*m1*yc2*s2 + l2*m2*xc3*c3 - l2*m2*yc3*s3);
    
    float M01 = Izz2+Izz3 + l2*l2*m2 + m1*(xc2*xc2+yc2*yc2) + m2*(xc3*xc3+yc3*yc3)
              + l1*m2*xc3*c23 + l1*l2*m2*c2 - l1*m2*yc3*s23
              + 2.0f*(l1*m1*xc2*c2 - l1*m1*yc2*s2 + l2*m2*xc3*c3 - l2*m2*yc3*s3);
    
    float M02 = Izz3 + m2*(xc3*xc3+yc3*yc3) + l1*m2*xc3*c23 - l1*m2*yc3*s23
              + l2*m2*xc3*c3 - l2*m2*yc3*s3;
    
    float M10 = M01, M20 = M02;
    float M11 = Izz2+Izz3 + l2*l2*m2 + m1*(xc2*xc2+yc2*yc2) + m2*(xc3*xc3+yc3*yc3)
              + 2.0f*(l2*m2*xc3*c3 - l2*m2*yc3*s3);
    float M12 = Izz3 + m2*(xc3*xc3+yc3*yc3) + l2*m2*xc3*c3 - l2*m2*yc3*s3;
    float M21 = M12;
    float M22 = Izz3 + m2*(xc3*xc3+yc3*yc3);
    
    // Coriolis terms
    float dq1=q_dot[0], dq2=q_dot[1], dq3=q_dot[2];
    
    float C00 = -dq3*m2*(l2*xc3*s3+l1*yc3*c23+l1*xc3*s23+l2*yc3*c3)
              - l1*dq2*(m1*yc2*c2+m1*xc2*s2+m2*yc3*c23+m2*xc3*s23+l2*m2*s2);
    float C01 = C00 - l1*dq1*(m1*yc2*c2+m1*xc2*s2+m2*yc3*c23+m2*xc3*s23+l2*m2*s2);
    float C02 = -m2*(dq1+dq2+dq3)*(l2*xc3*s3+l1*yc3*c23+l1*xc3*s23+l2*yc3*c3);
    float C10 = l1*dq1*(m1*yc2*c2+m1*xc2*s2+m2*yc3*c23+m2*xc3*s23+l2*m2*s2)
              - l2*dq3*m2*(yc3*c3+xc3*s3);
    float C11 = -l2*dq3*m2*(yc3*c3+xc3*s3);
    float C12 = -l2*m2*(yc3*c3+xc3*s3)*(dq1+dq2+dq3);
    float C20 = dq1*(l1*m2*yc3*c23+l1*m2*xc3*s23+l2*m2*yc3*c3+l2*m2*xc3*s3)
              + dq2*(l2*m2*yc3*c3+l2*m2*xc3*s3);
    float C21 = l2*m2*(dq1+dq2)*(yc3*c3+xc3*s3);
    float C22 = 0.0f;
    
    // PD control
    float v[3];
    for (int i = 0; i < 3; i++) {
        float e = qd[i] - q[i];
        float de = qd_dot[i] - q_dot[i];
        v[i] = qd_ddot[i] + ctc->gains.kd[i]*de + ctc->gains.kp[i]*e;
    }
    
    // M*v + C*q_dot
    float Mv[3] = {0,0,0}, Cq[3] = {0,0,0};
    float Mrow[3][3] = {{M00,M01,M02},{M10,M11,M12},{M20,M21,M22}};
    float Crow[3][3] = {{C00,C01,C02},{C10,C11,C12},{C20,C21,C22}};
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Mv[i] += Mrow[i][j] * v[j];
            Cq[i] += Crow[i][j] * q_dot[j];
        }
        float sign_qd = (q_dot[i] >= 0.0f) ? 1.0f : -1.0f;
        tau_out[i] = Mv[i] + Cq[i] + G_blend[i] 
                   + m->viscous_friction[i]*q_dot[i] 
                   + m->coulomb_friction[i]*sign_qd;
        tau_out[i] *= m->torque_scale;
        
        if (tau_out[i] > TORQUE_MAX_NM) tau_out[i] = TORQUE_MAX_NM;
        if (tau_out[i] < -TORQUE_MAX_NM) tau_out[i] = -TORQUE_MAX_NM;
    }
}

// ============================================================================
// Control State
// ============================================================================
typedef struct {
    float q_deg[3];
    float qdot_deg[3];
    SplineTrajectory traj[3];
    float traj_max_vel_deg_s;
    float startup_t_s;
    bool motion_active;
    float tau_joint[3];
    bool locked_axes[3];
    float gear_ratios[3];
    float motor_efficiency[3];
    CTC3DOF ctc;
} CtrlState;

extern CtrlState g_ctrl;

// ============================================================================
// API
// ============================================================================
void onBoardCtlInit(void);
void onBoardCtlUpdate(void);
void onBoardCtlSetTargetDeg(float q0, float q1, float q2);
void onBoardCtlHoldCurrent(void);
void onBoardCtlSetLockedAxes(bool lock0, bool lock1, bool lock2);
float jointToMotorTorque(int axis, float tau_joint);

#ifdef __cplusplus
}
#endif

#endif // ON_BOARD_CTL_H
