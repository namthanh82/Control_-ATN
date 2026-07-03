/**
 * ODrive CAN Protocol Header for ESP-IDF
 * Implements ODrive CAN Simple protocol v3.6
 */

#ifndef ODRIVE_CAN_H
#define ODRIVE_CAN_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ODrive CAN Command IDs (v3.6 protocol)
// ============================================================================
#define ODRIVE_CMD_HEARTBEAT             0x01
#define ODRIVE_CMD_GET_ENCODER_ESTIMATES 0x09
#define ODRIVE_CMD_SET_CONTROLLER_MODES  0x0B
#define ODRIVE_CMD_SET_INPUT_POS         0x0C
#define ODRIVE_CMD_SET_INPUT_TORQUE      0x0E
#define ODRIVE_CMD_SET_INPUT_TORQUE_RAMP_RATE 0x34
#define ODRIVE_CMD_SET_AXIS_REQUESTED_STATE 0x07
#define ODRIVE_CMD_CLEAR_ERRORS          0x18
#define ODRIVE_CMD_SAVE_EEPROM          0x16

// ============================================================================
// ODrive Axis States
// ============================================================================
#define AXIS_STATE_IDLE                  1
#define AXIS_STATE_STARTUP_SEQUENCE       2
#define AXIS_STATE_FULL_CALIBRATION_SEQUENCE 3
#define AXIS_STATE_MOTOR_CALIBRATION     4
#define AXIS_STATE_ENCODER_INDEX_SEARCH   6
#define AXIS_STATE_ENCODER_OFFSET_CALIBRATION 7
#define AXIS_STATE_CLOSED_LOOP_CONTROL   8
#define AXIS_STATE_LOCKIN_SPIN           10
#define AXIS_STATE_ENCODER_DIR_FIND       11

// ============================================================================
// ODrive Control Modes
// ============================================================================
#define CONTROL_MODE_VOLTAGE_CONTROL     0
#define CONTROL_MODE_TORQUE_CONTROL       1
#define CONTROL_MODE_VELOCITY_CONTROL     2
#define CONTROL_MODE_POSITION_CONTROL      3

// ============================================================================
// ODrive Input Modes
// ============================================================================
#define INPUT_MODE_INACTIVE              0
#define INPUT_MODE_PASSTHROUGH           1
#define INPUT_MODE_VEL_RAMP             2
#define INPUT_MODE_POS_FILTER           3
#define INPUT_MODE_MIX_CHANNELS         4
#define INPUT_MODE_TRAP_TRAJ            5
#define INPUT_MODE_TORQUE_RAMP          6
#define INPUT_MODE_MIRROR               7

// ============================================================================
// ODrive Error Codes
// ============================================================================
#define ODRIVE_ERROR_NONE               0x00
#define ODRIVE_ERROR_INVALID_STATE      0x01
#define ODRIVE_ERROR_ESTOP             0x02
#define ODRIVE_ERROR_HOMING_WITHOUT_SET_POS 0x04

// ============================================================================
// CAN Message Structures
// ============================================================================

typedef struct __attribute__((packed)) {
    uint8_t axis_error;
    uint8_t motor_error;
    uint8_t encoder_error;
    uint8_t sensorless_error;
    uint16_t axis_state;
    uint8_t requested_state;
    uint8_t trajectory_done_flag;
} Heartbeat_msg_t;

typedef struct __attribute__((packed)) {
    float Pos_Estimate;
    float Vel_Estimate;
} Get_Encoder_Estimates_msg_t;

typedef struct __attribute__((packed)) {
    int32_t Pos_Estimate;
    int32_t Vel_Estimate;
} Get_Encoder_Estimates_raw_msg_t;

typedef struct __attribute__((packed)) {
    int32_t Shadow_Count;
    int32_t Count_in_CPR;
} Get_Encoder_Counts_msg_t;

typedef struct __attribute__((packed)) {
    float Motor_Power;
    float Bus_Voltage;
    float Bus_Current;
} Get_Bus_Voltage_Current_msg_t;

typedef struct __attribute__((packed)) {
    float Iq_Setpoint;
    float Iq_Measured;
    float Vbus_Voltage;
} Get_Iq_msg_t;

typedef struct __attribute__((packed)) {
    float Temperature_Motor;
    float Temperature_Inverter;
} Get_Temperatures_msg_t;

// ============================================================================
// ODrive User Data Structure
// ============================================================================
typedef struct {
    Heartbeat_msg_t last_heartbeat;
    bool received_heartbeat;
    Get_Encoder_Estimates_msg_t last_feedback;
    bool received_feedback;
    Get_Encoder_Counts_msg_t last_counts;
    bool received_counts;
} ODriveUserData;

// ============================================================================
// Helper Functions
// ============================================================================

// Pack float into int32 (1 unit = 1/2048 revolution = ~0.00049 rad)
static inline int32_t float_to_can_float(float val) {
    return (int32_t)(val * 2048.0f);
}

// Unpack int32 from CAN to float (1 unit = 1/2048 revolution)
static inline float can_float_to_float(int32_t val) {
    return val / 2048.0f;
}

// Pack torque: 1 unit = 1mNm (torque_constant assumed to be 0.02 Nm/A)
static inline int32_t torque_to_can_torque(float torque_nm) {
    return (int32_t)(torque_nm * 1000.0f);
}

// Unpack torque from CAN
static inline float can_torque_to_torque(int32_t val) {
    return val / 1000.0f;
}

// Write float (IEEE754) to byte array (little-endian)
static inline void write_float_le(uint8_t* dst, float value) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.f = value;
    dst[0] = conv.u & 0xFF;
    dst[1] = (conv.u >> 8) & 0xFF;
    dst[2] = (conv.u >> 16) & 0xFF;
    dst[3] = (conv.u >> 24) & 0xFF;
}

// Read float (IEEE754) from byte array (little-endian)
static inline float read_float_le(const uint8_t* src) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.u = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24);
    return conv.f;
}

// Write uint32 to byte array (little-endian)
static inline void write_u32_le(uint8_t* dst, uint32_t value) {
    dst[0] = value & 0xFF;
    dst[1] = (value >> 8) & 0xFF;
    dst[2] = (value >> 16) & 0xFF;
    dst[3] = (value >> 24) & 0xFF;
}

// Read uint32 from byte array (little-endian)
static inline uint32_t read_u32_le(const uint8_t* src) {
    return src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24);
}

// Read int32 from byte array (little-endian)
static inline int32_t read_i32_le(const uint8_t* src) {
    return (int32_t)read_u32_le(src);
}

// Write int32 to byte array (little-endian)
static inline void write_i32_le(uint8_t* dst, int32_t value) {
    write_u32_le(dst, (uint32_t)value);
}

#ifdef __cplusplus
}
#endif

#endif // ODRIVE_CAN_H
