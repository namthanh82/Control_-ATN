// =============================================================================
//  ODriveCAN.h — Native ODrive CAN Protocol for ESP32-S3 TWAI
//  No external library required, pure ESP-IDF TWAI implementation
// =============================================================================
#pragma once

#include <cstdint>

// ============================================================================
// ODrive CAN Command IDs
// ============================================================================
#define ODRIVE_CMD_HEARTBEAT            0x01
#define ODRIVE_CMD_GET_ENCODER          0x09
#define ODRIVE_CMD_SET_AXIS_STATE       0x07
#define ODRIVE_CMD_SET_INPUT_TORQUE     0x10
#define ODRIVE_CMD_GET_ERROR            0x03
#define ODRIVE_CMD_SET_CONTROLLER_MODE  0x0B
#define ODRIVE_CMD_CLEAR_ERRORS         0x18
#define ODRIVE_CMD_SAVE_EEPROM         0x16
#define ODRIVE_CMD_GET_IQ               0x14
#define ODRIVE_CMD_GET_VBUS_VOLTAGE     0x17

// ============================================================================
// Axis States
// ============================================================================
#define AXIS_STATE_UNDEFINED            0
#define AXIS_STATE_IDLE                1
#define AXIS_STATE_STARTUP_SEQUENCE     2
#define AXIS_STATE_FULL_CALIBRATION    3
#define AXIS_STATE_MOTOR_CALIBRATION    4
#define AXIS_STATE_ENCODER_INDEX_SEARCH 6
#define AXIS_STATE_ENCODER_OFFSET_CALIBRATION 7
#define AXIS_STATE_CLOSED_LOOP_CONTROL  8
#define AXIS_STATE_LOCKIN_SPIN         10
#define AXIS_STATE_ENCODER_DIR_FIND    11

// ============================================================================
// Control Modes
// ============================================================================
#define CONTROL_MODE_VOLTAGE_CONTROL    0
#define CONTROL_MODE_TORQUE_CONTROL    1
#define CONTROL_MODE_VELOCITY_CONTROL  2

// ============================================================================
// Input Modes
// ============================================================================
#define INPUT_MODE_INACTIVE             0
#define INPUT_MODE_PASSTHROUGH          1
#define INPUT_MODE_VEL_RAMP             2
#define INPUT_MODE_TORQUE_RAMP          6
#define INPUT_MODE_MIRROR               4
#define INPUT_MODE_TRAP_TRAJ            5

// ============================================================================
// Error Codes
// ============================================================================
#define AXIS_ERROR_NONE                 0
#define AXIS_ERROR_INVALID_STATE        (1 << 0)
#define AXIS_ERROR_WATCHDOG_TIMER       (1 << 1)
#define AXIS_ERROR_MIN_ENDSTOP_APPLIED  (1 << 2)
#define AXIS_ERROR_MAX_ENDSTOP_APPLIED  (1 << 3)
#define AXIS_ERROR_ESTOP_REQUESTED      (1 << 4)
#define AXIS_ERROR_HOMING_WITHOUT_LIMITS (1 << 5)
#define AXIS_ERROR_OVERVOLTAGE         (1 << 6)
#define AXIS_ERROR_UNDER_VOLTAGE       (1 << 7)
#define AXIS_ERROR_ENCODDER_ERROR       (1 << 8)
#define AXIS_ERROR_BRAKE_RESISTOR_DISARMED (1 << 9)
#define AXIS_ERROR_MOTOR_DISARMED      (1 << 10)
#define AXIS_ERROR_SENSORLESS_ESTIMATOR_ERROR (1 << 11)
#define AXIS_ERROR_IDX_NOT_FOUND_YET   (1 << 12)
#define AXIS_ERROR_ENCODER_FAILED      (1 << 13)
#define AXIS_ERROR_CPR_FAILED          (1 << 14)
#define AXIS_ERROR_BRAKE_DEADTIME_VIOLATION (1 << 15)
#define AXIS_ERROR_BRAKE_DUTY_CYCLE_NAN (1 << 16)
#define AXIS_ERROR_CAN_RECEIVE         (1 << 17)
#define AXIS_ERROR_BRAKE_PRED_NOT_CLOSE (1 << 18)

// ============================================================================
// ODrive Message Structures
// ============================================================================
struct Heartbeat_msg_t {
    uint32_t Axis_State;
    uint32_t Procedure_Result;
    int32_t Trajectory_Done;
    uint32_t Axis_Error;
};

struct Get_Encoder_Estimates_msg_t {
    float Pos_Estimate;
    float Vel_Estimate;
};

struct ODriveError_msg_t {
    uint32_t Axis_Error;
    uint32_t Motor_Error;
    uint32_t Encoder_Error;
    uint32_t Controller_Error;
};

struct Get_Iq_msg_t {
    float Iq_Setpoint;
    float Iq_Measured;
};

struct Get_VBus_msg_t {
    float Vbus_Voltage;
};
