// =============================================================================
//  on_board_ctl.h — Trajectory + CTC + Control loop cho ESP32-S3
// =============================================================================
#pragma once

#include <cstdint>
#include <math.h>
#include <string.h>
#include "Arduino.h"

// ============================================================================
//  Constants — khớp với twai_serial_controller1.py
// ============================================================================
constexpr float     DEG2RAD       = 0.017453292519943295f;  // π/180
constexpr float     RAD2DEG       = 57.29577951308232f;
constexpr float     GRAVITY       = 9.81f;
constexpr float     TORQUE_MAX_NM = 1.5f;
constexpr uint32_t  CTRL_HZ       = 1000;
constexpr uint32_t  FB_HZ         = 100;

// Prismatic joint limits (350-450 mm)
constexpr float     PRISMATIC_MIN_MM = 350.0f;
constexpr float     PRISMATIC_MAX_MM = 450.0f;

// ============================================================================
//  EncoderDef — Quadrature A/B/Z encoder with PCNT
// ============================================================================
struct EncoderDef {
  int pinA;
  int pinB;
  int pinZ;
  int8_t id;
  int8_t direction;
  int32_t counts_per_rev;
};

// ============================================================================
//  CTC3Model — Full robot model parameters matching twai_serial_controller1.py
// ============================================================================
struct CTC3Model {
    float hip_mass1 = 2.928f;
    float hip_mass2 = 3.277f;
    float knee_mass1 = 2.898f;
    float knee_mass2 = 3.201f;
    float ankle_mass = 0.896f;
    
    float hip_x1 = 0.128f, hip_y1 = 0.02988f;
    float hip_x2_base = 0.28705f, hip_y2 = -0.00314f;
    float knee_x1 = 0.12843f, knee_y1 = 0.03047f;
    float knee_x2_base = 0.28561f, knee_y2 = -0.0032f;
    float ankle_x = 0.05723f, ankle_y = 0.0f;
    
    float j0_mass() const { return hip_mass1 + hip_mass2; }
    float j1_mass() const { return knee_mass1 + knee_mass2; }
    float j2_mass() const { return ankle_mass; }
    
    float hip_chung_x = 0.212f, hip_chung_y = 0.0124f;
    float knee_chung_x = 0.211f, knee_chung_y = 0.0128f;
    
    float hip_khau1 = 0.0f, hip_khau2 = 0.0f;
    float knee_khau1 = 0.0f, knee_khau2 = 0.0f;
    
    float hip_inertia = 0.064f;
    float knee_inertia = 0.058f;
    float ankle_inertia = 0.00274688527f;
    
    float hip_inertia1 = 0.02142f;
    float hip_inertia2 = 0.02679f;
    float knee_inertia1 = 0.02100f;
    float knee_inertia2 = 0.02642f;
    
    float hip_length = 0.35f;
    float knee_length = 0.35f;
    float ankle_length = 0.07f;
    
    float hip_motor_inertia = 0.002676f;
    float knee_motor_inertia = 0.000643f;
    float ankle_motor_inertia = 0.000643f;
    float hip_gear_ratio = 100.0f;
    float knee_gear_ratio = 50.0f;
    float ankle_gear_ratio = 50.0f;
    
    float gravity = GRAVITY;
    float coulomb_friction[3]  = {0.05f, 0.05f, 0.05f};
    float viscous_friction[3]  = {0.00276f, 0.00276f, 0.00276f};
    float torque_scale         = 1.0f;
    float model_home_deg[3]    = {0.0f, -90.0f, 0.0f};
    
    float delta_x1 = 0.0f;
    float delta_x2 = 0.0f;
    
    void update_com() {
        float hip_x2 = hip_x2_base + delta_x1;
        float knee_x2 = knee_x2_base + delta_x2;
        
        float m01 = hip_mass1, m02 = hip_mass2;
        float hip_total_mass = m01 + m02;
        hip_chung_x = (m01 * hip_x1 + m02 * hip_x2) / hip_total_mass;
        hip_chung_y = (m01 * hip_y1 + m02 * hip_y2) / hip_total_mass;
        
        float m11 = knee_mass1, m12 = knee_mass2;
        float knee_total_mass = m11 + m12;
        knee_chung_x = (m11 * knee_x1 + m12 * knee_x2) / knee_total_mass;
        knee_chung_y = (m11 * knee_y1 + m12 * knee_y2) / knee_total_mass;
        
        float hip_d1 = sqrtf(hip_x1*hip_x1 + hip_y1*hip_y1);
        float hip_d2 = sqrtf(hip_x2*hip_x2 + hip_y2*hip_y2);
        float knee_d1 = sqrtf(knee_x1*knee_x1 + knee_y1*knee_y1);
        float knee_d2 = sqrtf(knee_x2*knee_x2 + knee_y2*knee_y2);
        
        float hip_com_dist = sqrtf(hip_chung_x*hip_chung_x + hip_chung_y*hip_chung_y);
        float knee_com_dist = sqrtf(knee_chung_x*knee_chung_x + knee_chung_y*knee_chung_y);
        
        hip_inertia = hip_inertia1 + hip_inertia2 
                    + m01*hip_d1*hip_d1 + m02*hip_d2*hip_d2
                    - hip_total_mass*hip_com_dist*hip_com_dist;
        knee_inertia = knee_inertia1 + knee_inertia2
                     + m11*knee_d1*knee_d1 + m12*knee_d2*knee_d2
                     - knee_total_mass*knee_com_dist*knee_com_dist;
    }
};

struct Gains {
    float kp[3] = {3.0f, 6.0f, 3.0f};
    float kd[3] = {1.0f, 0.5f, 1.0f};
};

// ============================================================================
//  SplineTrajectory
// ============================================================================
class SplineTrajectory {
public:
    float max_jerk = 10.0f;
    float max_acc  = 5.0f;
    float start_p  = 0.0f;
    float end_p    = 0.0f;
    float direction = 1.0f;
    float total_time = 0.0f;
    float t1=0, t2=0, t3=0, t4=0, t5=0, t6=0, t7=0;
    float d1=0, d2=0, d3=0, d4=0, d5=0, d6=0;
    float v1=0, v2=0, v3=0, v4=0, v5=0, v6=0;
    float a_pk = 0.0f;

    void compute(float start, float end, float max_v) {
        start_p = start;
        end_p   = end;
        float dist = end - start;
        float abs_dist = fabsf(dist);
        direction = (dist >= 0.0f) ? 1.0f : -1.0f;

        if (abs_dist < 1e-6f) {
            total_time = 0.0f;
            t1=t2=t3=t4=t5=t6=t7=0;
            d1=d2=d3=d4=d5=d6=0;
            v1=v2=v3=v4=v5=v6=0;
            a_pk = 0;
            return;
        }

        float v_jerk_phase = (max_acc * max_acc) / max_jerk;
        float t_j=0, t_a=0, t_const=0, v_pk=0, a_pk_local=0;

        if (max_v < v_jerk_phase) {
            a_pk_local = sqrtf(max_v * max_jerk);
            t_j = a_pk_local / max_jerk;
            t_a = 0.0f;
            float d_req = max_v * (2.0f * t_j);
            if (abs_dist >= d_req) { t_const = (abs_dist - d_req) / max_v; v_pk = max_v; }
            else                   { t_const = 0; v_pk = a_pk_local * t_j; }
        } else {
            a_pk_local = max_acc;
            t_j = a_pk_local / max_jerk;
            t_a = (max_v - v_jerk_phase) / a_pk_local;
            float d_req = max_v * (2.0f * t_j + t_a);
            if (abs_dist >= d_req) { t_const = (abs_dist - d_req) / max_v; v_pk = max_v; }
            else {
                t_const = 0.0f;
                float d_acc_limit = 2.0f * (max_acc * max_acc * max_acc) / (max_jerk * max_jerk);
                if (abs_dist < d_acc_limit) {
                    a_pk_local = cbrtf(abs_dist * (max_jerk * max_jerk) / 2.0f);
                    t_j = a_pk_local / max_jerk;
                    t_a = 0.0f;
                    v_pk = a_pk_local * t_j;
                } else {
                    float c = 2.0f * (t_j * t_j) - (abs_dist / a_pk_local);
                    float delta = 9.0f * (t_j * t_j) - 4.0f * c;
                    if (delta < 0) delta = 0;
                    t_a = (-3.0f * t_j + sqrtf(delta)) / 2.0f;
                    v_pk = a_pk_local * t_j + a_pk_local * t_a;
                }
            }
        }

        t1=t3=t5=t7=t_j;
        t2=t6=t_a;
        t4=t_const;
        total_time = t1+t2+t3+t4+t5+t6+t7;
        a_pk = a_pk_local;

        float j = max_jerk;
        float a = a_pk_local;
        d1 = (1.0f/6.0f) * j * t1*t1*t1;
        v1 = 0.5f * j * t1*t1;
        d2 = d1 + v1*t2 + 0.5f*a*t2*t2;
        v2 = v1 + a*t2;
        v3 = v2 + a*t3 - 0.5f*j*t3*t3;
        d3 = d2 + v2*t3 + 0.5f*a*t3*t3 - (1.0f/6.0f)*j*t3*t3*t3;
        v4 = v3;
        d4 = d3 + v3*t4;
        v5 = v4 - 0.5f*j*t5*t5;
        d5 = d4 + v4*t5 - (1.0f/6.0f)*j*t5*t5*t5;
        v6 = v5 - a*t6;
        d6 = d5 + v5*t6 - 0.5f*a*t6*t6;
    }

    void reset() { total_time = 0; t1=t2=t3=t4=t5=t6=t7=0;
                   d1=d2=d3=d4=d5=d6=0; v1=v2=v3=v4=v5=v6=0; a_pk=0; }

    void desired(float t, float& pos, float& vel, float& acc) const {
        if (t <= 0.0f)               { pos = start_p; vel = 0; acc = 0; return; }
        if (t >= total_time || total_time <= 0.0f) { pos = end_p; vel = 0; acc = 0; return; }

        float j = max_jerk;
        float a = a_pk;
        float p=0, v=0, ac=0;

        if (t < t1) {
            float dt = t;
            ac = j*dt; v = 0.5f*j*dt*dt; p = (1.0f/6.0f)*j*dt*dt*dt;
        } else if (t < t1+t2) {
            float dt = t - t1;
            ac = a; v = v1 + a*dt; p = d1 + v1*dt + 0.5f*a*dt*dt;
        } else if (t < t1+t2+t3) {
            float dt = t - (t1+t2);
            ac = a - j*dt; v = v2 + a*dt - 0.5f*j*dt*dt;
            p = d2 + v2*dt + 0.5f*a*dt*dt - (1.0f/6.0f)*j*dt*dt*dt;
        } else if (t < t1+t2+t3+t4) {
            float dt = t - (t1+t2+t3);
            ac = 0; v = v3; p = d3 + v3*dt;
        } else if (t < t1+t2+t3+t4+t5) {
            float dt = t - (t1+t2+t3+t4);
            ac = -j*dt; v = v4 - 0.5f*j*dt*dt;
            p = d4 + v4*dt - (1.0f/6.0f)*j*dt*dt*dt;
        } else if (t < t1+t2+t3+t4+t5+t6) {
            float dt = t - (t1+t2+t3+t4+t5);
            ac = -a; v = v5 - a*dt; p = d5 + v5*dt - 0.5f*a*dt*dt;
        } else {
            float dt = t - (t1+t2+t3+t4+t5+t6);
            ac = -a + j*dt; v = v6 - a*dt + 0.5f*j*dt*dt;
            p = d6 + v6*dt - 0.5f*a*dt*dt + (1.0f/6.0f)*j*dt*dt*dt;
        }
        pos = start_p + p * direction;
        vel = v * direction;
        acc = ac * direction;
    }
};

// ============================================================================
//  CTC3DOF
// ============================================================================
class CTC3DOF {
public:
    CTC3Model model;
    Gains      gains;

    void compute(const float q[3], const float q_dot[3],
                 const float qd[3], const float qd_dot[3], const float qd_ddot[3],
                 float tau_out[3], float startup_t = 0.0f) {
        float g = model.gravity;

        float l1 = model.hip_length;
        float l2 = model.knee_length;
        
        model.update_com();
        
        float m0 = model.j0_mass();
        float m1 = model.j1_mass();
        float m2 = model.j2_mass();
        float xc1 = model.hip_chung_x;
        float yc1 = model.hip_chung_y;
        float xc2 = model.knee_chung_x;
        float yc2 = model.knee_chung_y;
        float xc3 = model.ankle_x;
        float yc3 = model.ankle_y;
        
        float c1   = cosf(q[0]);
        float s1   = sinf(q[0]);
        float c12  = cosf(q[0]+q[1]);
        float s12  = sinf(q[0]+q[1]);
        float c123 = cosf(q[0]+q[1]+q[2]);
        float s123 = sinf(q[0]+q[1]+q[2]);
        float G0 = g*(m1*(xc2*c12 + l1*c1 - yc2*s12)
                    + m2*(l2*c12 + l1*c1 + xc3*c123 - yc3*s123)
                    + m0*(xc1*c1 - yc1*s1));
        float G1 = g*(m1*(xc2*c12 - yc2*s12)
                    + m2*(l2*c12 + xc3*c123 - yc3*s123));
        float G2 = g*(m2*xc3*c123 - m2*yc3*s123);

        const float startup_duration = 0.8f;
        float w = startup_duration > 0 ? startup_t / startup_duration : 1.0f;
        if (w < 0) w = 0; if (w > 1) w = 1;

        float Gh0 = G0, Gh1 = G1, Gh2 = G2;
        float G_blend[3] = {
            Gh0 * (1.0f - w) + G0 * w,
            Gh1 * (1.0f - w) + G1 * w,
            Gh2 * (1.0f - w) + G2 * w,
        };

        float c2  = cosf(q[1]);
        float c3  = cosf(q[2]);
        float s2  = sinf(q[1]);
        float s3  = sinf(q[2]);
        float c23 = cosf(q[1]+q[2]);
        float s23 = sinf(q[1]+q[2]);
        float Izz1=model.hip_inertia, Izz2=model.knee_inertia, Izz3=model.ankle_inertia;
        
        float M00 = Izz1+Izz2+Izz3 + l1*l1*m1+l1*l1*m2+l2*l2*m2
                  + m0*(xc1*xc1)+m1*(xc2*xc2)+m2*(xc3*xc3)
                  + m0*(yc1*yc1)+m1*(yc2*yc2)+m2*(yc3*yc3)
                  + l1*m2*xc3*c23*2.0f + l1*l2*m2*c2*2.0f
                  - l1*m2*yc3*s23*2.0f + l1*m1*xc2*c2*2.0f
                  + l2*m2*xc3*c3*2.0f - l1*m1*yc2*s2*2.0f - l2*m2*yc3*s3*2.0f;
        float M01 = Izz2+Izz3+l2*l2*m2 + m1*(xc2*xc2)+m2*(xc3*xc3)
                  + m1*(yc2*yc2)+m2*(yc3*yc3)
                  + l1*m2*xc3*c23 + l1*l2*m2*c2 - l1*m2*yc3*s23
                  + l1*m1*xc2*c2 + l2*m2*xc3*c3*2.0f
                  - l1*m1*yc2*s2 - l2*m2*yc3*s3*2.0f;
        float M02 = Izz3+m2*(xc3*xc3)+m2*(yc3*yc3)
                  + l1*m2*xc3*c23 - l1*m2*yc3*s23
                  + l2*m2*xc3*c3 - l2*m2*yc3*s3;
        float M10=M01, M20=M02;
        float M11 = Izz2+Izz3+l2*l2*m2 + m1*(xc2*xc2)+m2*(xc3*xc3)
                  + m1*(yc2*yc2)+m2*(yc3*yc3)
                  + l2*m2*xc3*c3*2.0f - l2*m2*yc3*s3*2.0f;
        float M12 = Izz3+m2*(xc3*xc3)+m2*(yc3*yc3)
                  + l2*m2*xc3*c3 - l2*m2*yc3*s3;
        float M21=M12;
        float M22 = Izz3+m2*(xc3*xc3+yc3*yc3);

        float dq1=q_dot[0], dq2=q_dot[1], dq3=q_dot[2];
        float C00 = -dq3*m2*(l2*xc3*s3+l1*yc3*c23+l1*xc3*s23+l2*yc3*c3)
                  - l1*dq2*(m1*yc2*c2+m1*xc2*s2+m2*yc3*c23+m2*xc3*s23+l2*m2*s2);
        float C01 = -dq3*m2*(l2*xc3*s3+l1*yc3*c23+l1*xc3*s23+l2*yc3*c3)
                  - l1*dq1*(m1*yc2*c2+m1*xc2*s2+m2*yc3*c23+m2*xc3*s23+l2*m2*s2)
                  - l1*dq2*(m1*yc2*c2+m1*xc2*s2+m2*yc3*c23+m2*xc3*s23+l2*m2*s2);
        float C02 = -m2*(dq1+dq2+dq3)*(l2*xc3*s3+l1*yc3*c23+l1*xc3*s23+l2*yc3*c3);
        float C10 = l1*dq1*(m1*yc2*c2+m1*xc2*s2+m2*yc3*c23+m2*xc3*s23+l2*m2*s2)
                  - l2*dq3*m2*(yc3*c3+xc3*s3);
        float C11 = -l2*dq3*m2*(yc3*c3+xc3*s3);
        float C12 = -l2*m2*(yc3*c3+xc3*s3)*(dq1+dq2+dq3);
        float C20 = dq1*(l1*m2*yc3*c23+l1*m2*xc3*s23+l2*m2*yc3*c3+l2*m2*xc3*s3)
                  + dq2*(l2*m2*yc3*c3+l2*m2*xc3*s3);
        float C21 = l2*m2*(dq1+dq2)*(yc3*c3+xc3*s3);
        float C22 = 0.0f;

        float v[3];
        for (int i=0;i<3;i++) {
            float e   = qd[i]   - q[i];
            float de  = qd_dot[i] - q_dot[i];
            v[i] = qd_ddot[i] + gains.kd[i]*de + gains.kp[i]*e;
        }

        float Mv[3]={0,0,0}, Cq[3]={0,0,0};
        float Mrow[3][3] = {{M00,M01,M02},{M01,M11,M12},{M02,M12,M22}};
        float Crow[3][3] = {{C00,C01,C02},{C10,C11,C12},{C20,C21,C22}};
        for (int i=0;i<3;i++) {
            for (int j=0;j<3;j++) {
                Mv[i] += Mrow[i][j] * v[j];
                Cq[i] += Crow[i][j] * q_dot[j];
            }
            float sign_qd = (q_dot[i] >= 0.0f) ? 1.0f : -1.0f;
            tau_out[i] = Mv[i] + Cq[i] + G_blend[i]
                       + model.viscous_friction[i]*q_dot[i]
                       + model.coulomb_friction[i]*sign_qd;
            tau_out[i] *= model.torque_scale;
            if (tau_out[i] > TORQUE_MAX_NM)  tau_out[i] = TORQUE_MAX_NM;
            if (tau_out[i] < -TORQUE_MAX_NM) tau_out[i] = -TORQUE_MAX_NM;
        }
    }
};

// ============================================================================
//  Control state
// ============================================================================
struct CtrlState {
    volatile float q_deg[3]      = {0,0,0};
    volatile float qdot_deg[3]   = {0,0,0};

    SplineTrajectory traj[3];
    float traj_max_vel_deg_s = 25.0f;
    float startup_t_s        = 0.0f;
    bool  motion_active      = false;

    float tau_joint[3]        = {0,0,0};
    bool  locked_axes[3]      = {false, false, false};

    float gear_ratios[3]      = {100.0f, 50.0f, 50.0f};
    float motor_efficiency[3] = {1.0f, 1.0f, 1.0f};

    volatile float prismatic_hip_mm = 350.0f;
    volatile float prismatic_knee_mm = 350.0f;

    CTC3DOF ctc;
};

extern CtrlState g_ctrl;

// ============================================================================
//  API
// ============================================================================
void onBoardCtlInit();
void onBoardCtlUpdate();  // compute CTC torque (call in main loop)
void onBoardCtlSetTargetDeg(float q0, float q1, float q2);
void onBoardCtlHoldCurrent();
void onBoardCtlSetKpKd(float kp0, float kp1, float kp2,
                       float kd0, float kd1, float kd2);
void onBoardCtlSetPrismatic(float hip_mm, float knee_mm);
float jointToMotorTorque(int axis, float tau_joint);
void onBoardCtlGetPrismatic(float& hip_mm, float& knee_mm);
void onBoardCtlSetLockedAxes(bool lock0, bool lock1, bool lock2);
