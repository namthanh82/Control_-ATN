/**
 * ODrive CAN Protocol Definitions - ESP-IDF Version
 * ODrive CAN bus protocol constants and helper functions
 */

#ifndef ODRIVE_CAN_H
#define ODRIVE_CAN_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ODrive CAN Command IDs
// ============================================================================
#define ODRIVE_CMD_HEARTBEAT              0x01
#define ODRIVE_CMD_GET_ENCODER_ESTIMATES 0x09
#define ODRIVE_CMD_SET_INPUT_POS         0x0C
#define ODRIVE_CMD_SET_INPUT_TORQUE       0x0E
#define ODRIVE_CMD_SET_CONTROLLER_MODES  0x0B
#define ODRIVE_CMD_SET_AXIS_REQUESTED_STATE 0x07
#define ODRIVE_CMD_CLEAR_ERRORS          0x18
#define ODRIVE_CMD_SET_PROPERTY          0x05  // Set_Property: writes runtime config
// NOTE: 0x10 (Set_Input_Torque_Ramp_Rate) is NOT in ODrive 0.6.11 - use
// Set_Property (0x05) with endpoint 0x015 (torque_ramp_rate) instead.

// ============================================================================
// ODrive Property Endpoint IDs (for Set_Property cmd 0x05)
// Endpoint format: 16-bit ID = (property_id << 5) | type_id
//   type_id: 1=float, 2=int32, 3=bool
// Property IDs are in ODrive firmware (Moteus-style flat addressing).
// Reference: ODriveArduino/ODriveCAN.cpp setProperty() lookup.
// ============================================================================
#define ODRIVE_PROP_TORQUE_RAMP_RATE      0x015  // controller.config.torque_ramp_rate (float)
#define ODRIVE_PROP_CONTROL_MODE          0x00B  // controller.config.control_mode (int32)
#define ODRIVE_PROP_INPUT_MODE            0x00C  // controller.config.input_mode (int32)
#define ODRIVE_PROP_POS_GAIN              0x01A  // controller.config.pos_gain (float)
#define ODRIVE_PROP_VEL_GAIN              0x01B  // controller.config.vel_gain (float)
#define ODRIVE_PROP_VEL_INTEGRATOR_GAIN   0x01C  // controller.config.vel_integrator_gain (float)
#define ODRIVE_PROP_VEL_LIMIT             0x00F  // controller.config.vel_limit (float)
#define ODRIVE_PROP_CURRENT_LIMIT         0x010  // motor.config.current_lim (float)

// ============================================================================
// ODrive Axis States
// ============================================================================
#define AXIS_STATE_IDLE                1
#define AXIS_STATE_STARTUP_SEQUENCE    2
#define AXIS_STATE_FULL_CALIBRATION_SEQUENCE 3
#define AXIS_STATE_MOTOR_CALIBRATION   4
#define AXIS_STATE_SENSORLESS_CONTROL  5
#define AXIS_STATE_ENCODER_INDEX_SEARCH 6
#define AXIS_STATE_ENCODER_OFFSET_CALIBRATION 7
#define AXIS_STATE_CLOSED_LOOP_CONTROL 8
#define AXIS_STATE_LOCKIN_SPIN        10
#define AXIS_STATE_ESTIMATE_INDEX      11

// ============================================================================
// ODrive Control Modes
// ============================================================================
#define CONTROL_MODE_VOLTAGE_CONTROL    0
#define CONTROL_MODE_TORQUE_CONTROL    1
#define CONTROL_MODE_VELOCITY_CONTROL  2
#define CONTROL_MODE_POSITION_CONTROL  3

// ============================================================================
// ODrive Input Modes
// ============================================================================
#define INPUT_MODE_INACTIVE            0
#define INPUT_MODE_PASSTHROUGH        1
#define INPUT_MODE_VEL_RAMP           2
#define INPUT_MODE_POS_FILTER         3
#define INPUT_MODE_MIX_CHANNELS       4
#define INPUT_MODE_TRAP_TRAJ          5
#define INPUT_MODE_TORQUE_RAMP        6
#define INPUT_MODE_MIRROR             7

// ============================================================================
// ODrive Error Flags
// ============================================================================
#define ODRIVE_ERROR_NONE             0x00
#define ODRIVE_ERROR_INVALID_STATE    0x01
#define ODRIVE_ERROR_ESTOP            0x02
#define ODRIVE_ERROR_HOMING_WITHOUT_INDEX 0x10000

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Write 32-bit integer in little-endian format
 */
static inline void write_i32_le(uint8_t* buf, int32_t val) {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
    buf[2] = (uint8_t)((val >> 16) & 0xFF);
    buf[3] = (uint8_t)((val >> 24) & 0xFF);
}

/**
 * Read 32-bit integer from little-endian format
 */
static inline int32_t read_i32_le(const uint8_t* buf) {
    return (int32_t)(
        ((uint32_t)buf[0]) |
        ((uint32_t)buf[1] << 8) |
        ((uint32_t)buf[2] << 16) |
        ((uint32_t)buf[3] << 24)
    );
}

/**
 * Write float in little-endian format (IEEE 754)
 */
static inline void write_float_le(uint8_t* buf, float val) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.f = val;
    buf[0] = (uint8_t)(conv.u & 0xFF);
    buf[1] = (uint8_t)((conv.u >> 8) & 0xFF);
    buf[2] = (uint8_t)((conv.u >> 16) & 0xFF);
    buf[3] = (uint8_t)((conv.u >> 24) & 0xFF);
}

/**
 * Read float from little-endian format (IEEE 754)
 */
static inline float read_float_le(const uint8_t* buf) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.u = ((uint32_t)buf[0]) |
             ((uint32_t)buf[1] << 8) |
             ((uint32_t)buf[2] << 16) |
             ((uint32_t)buf[3] << 24);
    return conv.f;
}

/**
 * Build CAN ID from node ID and command
 */
static inline uint32_t make_can_id(uint8_t node_id, uint8_t cmd_id) {
    return ((uint32_t)node_id << 5) | (cmd_id & 0x1F);
}

/**
 * Extract node ID from CAN ID
 */
static inline uint8_t get_node_id(uint32_t can_id) {
    return (uint8_t)((can_id >> 5) & 0x3F);
}

/**
 * Extract command ID from CAN ID
 */
static inline uint8_t get_cmd_id(uint32_t can_id) {
    return (uint8_t)(can_id & 0x1F);
}

// ============================================================================
// ODrive User Data (for parsing feedback)
// ============================================================================
typedef struct {
    // Heartbeat
    uint8_t error;
    uint8_t motor_error;
    uint8_t encoder_error;
    uint16_t axis_state;
    uint8_t requested_state;
    bool received_heartbeat;
    
    // Encoder estimates
    float pos_estimate;
    float vel_estimate;
    bool received_feedback;
} ODriveUserData;

#ifdef __cplusplus
}
#endif

#endif // ODRIVE_CAN_H
