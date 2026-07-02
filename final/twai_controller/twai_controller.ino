#include <ODriveUART.h>
#include <ODriveCAN.h>

// =============================================================================
//  twai_controller_native.ino — ESP32-S3 Native TWAI (CAN) cho ODrive
//  Không cần ODriveArduino library, dùng trực tiếp ESP-IDF TWAI API
// =============================================================================
#include "on_board_ctl.h"
#include "ODriveCAN.h"
#include "driver/gpio.h"
#include "driver/twai.h"
#include "esp_attr.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// ============================================================================
// External Quadrature Encoder Configuration (A/B/Z)
// ============================================================================
constexpr int32_t ENCODER_CPR = 4096;
constexpr int32_t ENCODER_CPR_X4 = ENCODER_CPR;  // 4096 = x4 decoded counts/rev (encoder 1024 PPR)

EncoderDef enc0 = {4, 5, 6, 0, 1, ENCODER_CPR_X4};
EncoderDef enc1 = {7, 8, 9, 1, 1, ENCODER_CPR_X4};
EncoderDef enc2 = {10, 11, 12, 2, 1, ENCODER_CPR_X4};

struct EncoderState {
    volatile int32_t count;
    uint8_t last_AB;
    uint8_t pinA;
    uint8_t pinB;
};

static EncoderState encoder_states[3];

void IRAM_ATTR encoder_isr(void* arg) {
    EncoderState* enc = (EncoderState*)arg;
    uint8_t a = gpio_get_level((gpio_num_t)enc->pinA) ? 1 : 0;
    uint8_t b = gpio_get_level((gpio_num_t)enc->pinB) ? 1 : 0;
    uint8_t AB = (a << 0) | (b << 1);
    
    switch ((enc->last_AB << 2) | AB) {
        case 0b0001: case 0b0111: case 0b1110: case 0b1000: enc->count++; break;
        case 0b0010: case 0b1011: case 0b1101: case 0b0100: enc->count--; break;
    }
    enc->last_AB = AB;
}

bool setupQuadratureEncoder_x4(int pinA, int pinB, EncoderState* enc_state) {
    enc_state->pinA = pinA;
    enc_state->pinB = pinB;
    enc_state->count = 0;
    enc_state->last_AB = (gpio_get_level((gpio_num_t)pinA) ? 1 : 0) | 
                         (gpio_get_level((gpio_num_t)pinB) ? 2 : 0);
    
    gpio_config_t io_conf = {};
    io_conf.pin_bit_mask = (1ULL << pinA) | (1ULL << pinB);
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.intr_type = GPIO_INTR_ANYEDGE;
    
    gpio_config(&io_conf);
    gpio_isr_handler_add((gpio_num_t)pinA, encoder_isr, enc_state);
    gpio_isr_handler_add((gpio_num_t)pinB, encoder_isr, enc_state);
    
    return true;
}

void setupAllEncoders() {
    gpio_install_isr_service(ESP_INTR_FLAG_IRAM);
    setupQuadratureEncoder_x4(enc0.pinA, enc0.pinB, &encoder_states[0]);
    setupQuadratureEncoder_x4(enc1.pinA, enc1.pinB, &encoder_states[1]);
    setupQuadratureEncoder_x4(enc2.pinA, enc2.pinB, &encoder_states[2]);
    pinMode(enc0.pinZ, INPUT_PULLUP);
    pinMode(enc1.pinZ, INPUT_PULLUP);
    pinMode(enc2.pinZ, INPUT_PULLUP);
    Serial.printf("INFO: Encoders x4 initialized (CPR=%d)\n", ENCODER_CPR_X4);
}

int32_t readEncoderCount(const EncoderDef& enc) {
    return encoder_states[enc.id].count * enc.direction;
}

float encoderCountToDeg(const EncoderDef& enc, int32_t count, float zero_deg, int32_t zero_count) {
    float rev = (float)(count - zero_count) / (float)enc.counts_per_rev;
    float deg = rev * 360.0f;
    deg += zero_deg;
    return deg;
}

// ============================================================================
// ODrive Protocol Constants (see ODriveCAN.h for full definitions)
// ============================================================================
#define ODRV0_NODE_ID 0   // ODrive #1
#define ODRV1_NODE_ID 1   // ODrive #2
#define ODRV2_NODE_ID 2   // ODrive #3

#define ODRIVE_CMD_SET_AXIS_STATE       0x07
#define ODRIVE_CMD_SET_INPUT_TORQUE     0x10
#define ODRIVE_CMD_SET_CONTROLLER_MODE  0x0B
#define ODRIVE_CMD_CLEAR_ERRORS         0x18
#define ODRIVE_CMD_GET_ENCODER          0x09
#define ODRIVE_CMD_HEARTBEAT            0x01

struct ODriveUserData {
    Heartbeat_msg_t last_heartbeat;
    bool received_heartbeat = false;
    Get_Encoder_Estimates_msg_t last_feedback;
    bool received_feedback = false;
};

static ODriveUserData odrv_data[3];
static bool closed_loop_requested = false;

static float ramp_torque0_nm = 0.0f;
static float ramp_torque1_nm = 0.0f;
static float ramp_torque2_nm = 0.0f;
static float last_torque0_nm = 0.0f;
static float last_torque1_nm = 0.0f;
static float last_torque2_nm = 0.0f;

static float last_pos0 = 0.0f, last_pos1 = 0.0f, last_pos2 = 0.0f;
static bool has_pos0 = false, has_pos1 = false, has_pos2 = false;
static float joint_deg0 = 0.0f, joint_deg1 = 0.0f, joint_deg2 = 0.0f;
static int32_t cnt0 = 0, cnt1 = 0, cnt2 = 0;

static unsigned long last_torque_tx_ms = 0;
static unsigned long last_fb_ms = 0;
#define TORQUE_TX_INTERVAL_MS 10
#define TORQUE_MAX_NM 2.0f

static float current_vel_deg_s[3] = {0.0f, 0.0f, 0.0f};
static unsigned long last_vel_calc_ms = 0;
static float last_joint_deg[3] = {0.0f, 0.0f, 0.0f};
static unsigned long torque_cmd_count = 0;

void applyTorqueRamp() { torque_cmd_count++; }

// ============================================================================
// TWAI Hardware Configuration - Legacy Arduino API
// ============================================================================
#include "driver/twai.h"
#include "esp_intr_alloc.h"

#define ESP32_TWAI_TX_PIN 13
#define ESP32_TWAI_RX_PIN 14

static bool twai_initialized = false;
static uint32_t can_tx_count = 0;
static uint32_t can_tx_fail = 0;
static uint32_t can_rx_count = 0;

bool setupCan() {
    twai_general_config_t g_config = {
        .mode = TWAI_MODE_NORMAL,
        .tx_io = (gpio_num_t)ESP32_TWAI_TX_PIN,
        .rx_io = (gpio_num_t)ESP32_TWAI_RX_PIN,
        .clkout_io = GPIO_NUM_NC,
        .bus_off_io = GPIO_NUM_NC,
        .tx_queue_len = 64,      // Increased queue
        .rx_queue_len = 256,
        .alerts_enabled = TWAI_ALERT_ALL,
        .clkout_divider = 0,
    };
    
    twai_timing_config_t t_config = {
        .brp = 8,                // 250kbps @ 80MHz APB
        .tseg_1 = 15,
        .tseg_2 = 4,
        .sjw = 2,
        .triple_sampling = false,
    };
    
    twai_filter_config_t f_config = {
        .acceptance_code = 0,
        .acceptance_mask = 0x7FF,  // Accept all standard IDs
        .single_filter = true,
    };

    esp_err_t err = twai_driver_install(&g_config, &t_config, &f_config);
    if (err != ESP_OK) {
        Serial.printf("ERROR: TWAI install failed: %s\n", esp_err_to_name(err));
        return false;
    }

    err = twai_start();
    if (err != ESP_OK) {
        Serial.printf("ERROR: TWAI start failed: %s\n", esp_err_to_name(err));
        twai_driver_uninstall();
        return false;
    }

    twai_initialized = true;
    Serial.println("INFO: TWAI initialized @ 250kbps (legacy API)");
    return true;
}

void serviceCanHealth() {
    uint32_t alerts = 0;
    twai_read_alerts(&alerts, pdMS_TO_TICKS(0));
    
    if (alerts & TWAI_ALERT_BUS_OFF) {
        Serial.println("WARN: CAN Bus Off - recovering...");
        twai_initiate_recovery();
    }
    if (alerts & TWAI_ALERT_BUS_RECOVERED) {
        twai_start();
        Serial.println("INFO: CAN Bus recovered");
    }
    if (alerts & TWAI_ALERT_TX_QUEUE_FULL) {
        Serial.println("WARN: TX queue full");
    }
    if (alerts & TWAI_ALERT_RX_QUEUE_FULL) {
        Serial.println("WARN: RX queue full");
    }
}

// ============================================================================
// ODrive CAN Message Sending
// ============================================================================

bool sendCanMessage(uint32_t can_id, const uint8_t* data, uint8_t len) {
    if (!twai_initialized) return false;
    
    twai_message_t msg;
    msg.identifier = can_id;
    msg.flags = TWAI_MSG_FLAG_NONE;
    msg.data_length_code = len;
    for (int i = 0; i < len && i < 8; i++) {
        msg.data[i] = data[i];
    }
    
    // Non-blocking TX with timeout
    esp_err_t err = twai_transmit(&msg, pdMS_TO_TICKS(10));
    
    if (err == ESP_OK) {
        can_tx_count++;
        return true;
    } else if (err == ESP_ERR_INVALID_STATE) {
        // TX queue full - wait and retry
        delay(1);
        err = twai_transmit(&msg, pdMS_TO_TICKS(20));
        if (err == ESP_OK) {
            can_tx_count++;
            return true;
        }
    }
    
    can_tx_fail++;
    if (can_tx_fail <= 5) {
        Serial.printf("TX_FAIL id=0x%03X err=%s\n", can_id, esp_err_to_name(err));
    }
    return false;
}

void sendAxisState(uint8_t node_id, uint32_t axis_state) {
    uint8_t data[4];
    data[0] = axis_state & 0xFF;
    data[1] = (axis_state >> 8) & 0xFF;
    data[2] = (axis_state >> 16) & 0xFF;
    data[3] = (axis_state >> 24) & 0xFF;
    
    // v3.6 format: node_id ở bit 5-9, cmd ở bit 0-4
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_SET_AXIS_STATE;
    Serial.printf("TX SET_STATE id=0x%03X node=%d state=%d\n", can_id, node_id, axis_state);
    sendCanMessage(can_id, data, 4);
}

void sendInputTorque(uint8_t node_id, float torque) {
    uint8_t data[4];
    int32_t torque_int = (int32_t)(torque * 1000.0f);  // 1mNm resolution
    data[0] = torque_int & 0xFF;
    data[1] = (torque_int >> 8) & 0xFF;
    data[2] = (torque_int >> 16) & 0xFF;
    data[3] = (torque_int >> 24) & 0xFF;
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_SET_INPUT_TORQUE;
    sendCanMessage(can_id, data, 4);
}

void sendControllerMode(uint8_t node_id, int32_t control_mode, int32_t input_mode) {
    uint8_t data[8];
    data[0] = control_mode & 0xFF;
    data[1] = (control_mode >> 8) & 0xFF;
    data[2] = (control_mode >> 16) & 0xFF;
    data[3] = (control_mode >> 24) & 0xFF;
    data[4] = input_mode & 0xFF;
    data[5] = (input_mode >> 8) & 0xFF;
    data[6] = (input_mode >> 16) & 0xFF;
    data[7] = (input_mode >> 24) & 0xFF;
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_SET_CONTROLLER_MODE;
    Serial.printf("TX SET_CTRL mode=%d input=%d node=%d id=0x%03X\n", 
                   control_mode, input_mode, node_id, can_id);
    delay(10);  // Small delay between messages
    sendCanMessage(can_id, data, 8);
}

void sendGetEncoderEstimates(uint8_t node_id) {
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_GET_ENCODER;
    sendCanMessage(can_id, nullptr, 0);
}

void sendClearErrors(uint8_t node_id) {
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_CLEAR_ERRORS;
    sendCanMessage(can_id, nullptr, 0);
}

void sendSaveConfig(uint8_t node_id) {
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_SAVE_EEPROM;
    sendCanMessage(can_id, nullptr, 0);
    Serial.printf("TX SAVE_CONFIG node=%d\n", node_id);
}

// ============================================================================
// ODrive CAN Message Receiving
// ============================================================================
void processCanRx() {
    twai_message_t rx_msg;
    esp_err_t err = twai_receive(&rx_msg, pdMS_TO_TICKS(0));
    
    if (err == ESP_OK) {
        can_rx_count++;
        uint8_t node_id = (rx_msg.identifier >> 5) & 0x1F;
        uint8_t cmd = rx_msg.identifier & 0x1F;
        
        // Debug: print first 30 CAN frames
        if (can_rx_count <= 30) {
            Serial.printf("CAN id=0x%03X node=%d cmd=%d dlc=%d [", 
                rx_msg.identifier, node_id, cmd, rx_msg.data_length_code);
            for (int i = 0; i < rx_msg.data_length_code && i < 8; i++) {
                Serial.printf("%02X ", rx_msg.data[i]);
            }
            Serial.println("]");
        }
        
        switch (cmd) {
            case ODRIVE_CMD_HEARTBEAT: {
                if (rx_msg.data_length_code >= 6) {
                    uint32_t axis_state = rx_msg.data[4] | (rx_msg.data[5] << 8);
                    int32_t trajectory_done = rx_msg.data[6] | (rx_msg.data[7] << 8);
                    odrv_data[node_id].last_heartbeat.Axis_State = axis_state;
                    odrv_data[node_id].last_heartbeat.Trajectory_Done = trajectory_done;
                    odrv_data[node_id].received_heartbeat = true;
                }
                break;
            }
            case ODRIVE_CMD_GET_ENCODER: {
                if (rx_msg.data_length_code >= 8) {
                    int32_t pos = rx_msg.data[0] | (rx_msg.data[1] << 8) | 
                                  (rx_msg.data[2] << 16) | (rx_msg.data[3] << 24);
                    int32_t vel = rx_msg.data[4] | (rx_msg.data[5] << 8) | 
                                  (rx_msg.data[6] << 16) | (rx_msg.data[7] << 24);
                    odrv_data[node_id].last_feedback.Pos_Estimate = (float)pos / 2048.0f;
                    odrv_data[node_id].last_feedback.Vel_Estimate = (float)vel / 2048.0f * 100.0f;
                    odrv_data[node_id].received_feedback = true;
                    
                    if (node_id == 0) { last_pos0 = odrv_data[0].last_feedback.Pos_Estimate; has_pos0 = true; }
                    if (node_id == 1) { last_pos1 = odrv_data[1].last_feedback.Pos_Estimate; has_pos1 = true; }
                    if (node_id == 2) { last_pos2 = odrv_data[2].last_feedback.Pos_Estimate; has_pos2 = true; }
                }
                break;
            }
        }
    }
    // Don't log ESP_ERR_TIMEOUT - that's normal when no messages
}

// ============================================================================
// Heartbeat Wait
// ============================================================================
bool waitHeartbeat(uint8_t node_id, const char* label, uint32_t timeout_ms) {
    Serial.print("INFO: Waiting heartbeat ");
    Serial.println(label);
    unsigned long t0 = millis();
    odrv_data[node_id].received_heartbeat = false;
    
    while (!odrv_data[node_id].received_heartbeat) {
        processCanRx();
        if ((millis() - t0) > timeout_ms) {
            Serial.print("ERROR: heartbeat timeout ");
            Serial.println(label);
            return false;
        }
        delay(2);
    }
    Serial.printf("INFO: Heartbeat received from %s (state=%d)\n", 
                  label, odrv_data[node_id].last_heartbeat.Axis_State);
    return true;
}

// ============================================================================
// Control Functions
// ============================================================================
void setClosedLoop(bool enable) {
    uint32_t target = enable ? AXIS_STATE_CLOSED_LOOP_CONTROL : AXIS_STATE_IDLE;
    sendAxisState(ODRV0_NODE_ID, target);
    sendAxisState(ODRV1_NODE_ID, target);
    sendAxisState(ODRV2_NODE_ID, target);
    closed_loop_requested = enable;
    
    if (enable) {
        ramp_torque0_nm = ramp_torque1_nm = ramp_torque2_nm = 0.0f;
        Serial.println("INFO: Closed loop enabled, torque reset to 0");
    } else {
        ramp_torque0_nm = ramp_torque1_nm = ramp_torque2_nm = 0.0f;
        Serial.println("INFO: IDLE mode");
    }
}

void setTorqueMode() {
    sendControllerMode(ODRV0_NODE_ID, CONTROL_MODE_TORQUE_CONTROL, INPUT_MODE_TORQUE_RAMP);
    sendControllerMode(ODRV1_NODE_ID, CONTROL_MODE_TORQUE_CONTROL, INPUT_MODE_TORQUE_RAMP);
    sendControllerMode(ODRV2_NODE_ID, CONTROL_MODE_TORQUE_CONTROL, INPUT_MODE_TORQUE_RAMP);
    Serial.println("INFO: controller mode=TORQUE/TORQUE_RAMP");
}

void clearBothErrors() {
    sendClearErrors(ODRV0_NODE_ID);
    sendClearErrors(ODRV1_NODE_ID);
    sendClearErrors(ODRV2_NODE_ID);
    Serial.println("INFO: clearErrors sent");
}

void setHomeIndexSearch() {
    sendAxisState(ODRV0_NODE_ID, AXIS_STATE_IDLE);
    sendAxisState(ODRV1_NODE_ID, AXIS_STATE_IDLE);
    sendAxisState(ODRV2_NODE_ID, AXIS_STATE_IDLE);
    delay(100);
    sendAxisState(ODRV0_NODE_ID, AXIS_STATE_ENCODER_INDEX_SEARCH);
    sendAxisState(ODRV1_NODE_ID, AXIS_STATE_ENCODER_INDEX_SEARCH);
    sendAxisState(ODRV2_NODE_ID, AXIS_STATE_ENCODER_INDEX_SEARCH);
    Serial.println("INFO: state request=INDEX_SEARCH");
}

// ============================================================================
// Serial Command Processing
// ============================================================================
void processSerialCommand();

// Use g_ctrl.motion_active directly (defined in on_board_ctl.h)

void onBoardCtlHoldCurrent() __attribute__((weak));
void onBoardCtlHoldCurrent() {
    g_ctrl.motion_active = false;
    ramp_torque0_nm = ramp_torque1_nm = ramp_torque2_nm = 0.0f;
}

// ============================================================================
// Setup & Loop
// ============================================================================
void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    Serial.println("\n\n=== ESP32-S3 TWAI ODrive Controller ===");
    Serial.println("Using Native ESP-IDF TWAI API (no external library)");
    
    setupAllEncoders();
    onBoardCtlInit();
    
    if (!setupCan()) {
        Serial.println("ERROR: CAN init failed - looping forever");
        while (true) delay(1000);
    }
    
    // Request heartbeat from all nodes
    delay(100);
    
    if (!waitHeartbeat(ODRV0_NODE_ID, "node0", 7000)) {
        Serial.println("WARN: node0 not responding");
    }
    if (!waitHeartbeat(ODRV1_NODE_ID, "node1", 7000)) {
        Serial.println("WARN: node1 not responding");
    }
    if (!waitHeartbeat(ODRV2_NODE_ID, "node2", 7000)) {
        Serial.println("WARN: node2 not responding");
    }
    
    // Step 1: Force torque=0 while IDLE to clear any residual torque
    for (int i = 0; i < 20; i++) {
        sendAxisState(ODRV0_NODE_ID, AXIS_STATE_IDLE);
        sendAxisState(ODRV1_NODE_ID, AXIS_STATE_IDLE);
        sendAxisState(ODRV2_NODE_ID, AXIS_STATE_IDLE);
        sendInputTorque(ODRV0_NODE_ID, 0.0f);
        sendInputTorque(ODRV1_NODE_ID, 0.0f);
        sendInputTorque(ODRV2_NODE_ID, 0.0f);
        delay(5);
    }
    
    // Step 2: Clear errors and set torque mode
    Serial.println("INFO: Clearing errors...");
    clearBothErrors();
    delay(100);
    
    Serial.println("INFO: Setting torque mode...");
    setTorqueMode();
    delay(100);
    
    Serial.println("INFO: Saving config...");
    sendSaveConfig(ODRV0_NODE_ID);
    delay(50);
    sendSaveConfig(ODRV1_NODE_ID);
    delay(50);
    sendSaveConfig(ODRV2_NODE_ID);
    delay(300);
    
    // Step 3: Reset torque one more time before closing loop
    for (int i = 0; i < 5; i++) {
        sendInputTorque(ODRV0_NODE_ID, 0.0f);
        delay(5);
        sendInputTorque(ODRV1_NODE_ID, 0.0f);
        delay(5);
        sendInputTorque(ODRV2_NODE_ID, 0.0f);
        delay(10);
    }
    
    // Step 4: Now enter closed loop
    setClosedLoop(true);
    
    Serial.println("READY");
}

void loop() {
    processSerialCommand();
    processCanRx();
    serviceCanHealth();
    
    unsigned long now_tx = millis();
    
    if (g_ctrl.motion_active) {
        onBoardCtlUpdate();
    }
    
    if (now_tx - last_torque_tx_ms >= TORQUE_TX_INTERVAL_MS) {
        last_torque_tx_ms = now_tx;
        
    if (g_ctrl.motion_active) {
        ramp_torque0_nm = g_ctrl.tau_joint[0];
        ramp_torque1_nm = g_ctrl.tau_joint[1];
        ramp_torque2_nm = g_ctrl.tau_joint[2];
        
        if (g_ctrl.locked_axes[0]) ramp_torque0_nm = 0.0f;
        if (g_ctrl.locked_axes[1]) ramp_torque1_nm = 0.0f;
        if (g_ctrl.locked_axes[2]) ramp_torque2_nm = 0.0f;
    } else {
            applyTorqueRamp();
        }
        
        ramp_torque0_nm = constrain(ramp_torque0_nm, -TORQUE_MAX_NM, TORQUE_MAX_NM);
        ramp_torque1_nm = constrain(ramp_torque1_nm, -TORQUE_MAX_NM, TORQUE_MAX_NM);
        ramp_torque2_nm = constrain(ramp_torque2_nm, -TORQUE_MAX_NM, TORQUE_MAX_NM);
        
        sendInputTorque(ODRV0_NODE_ID, ramp_torque0_nm);
        sendInputTorque(ODRV1_NODE_ID, ramp_torque1_nm);
        sendInputTorque(ODRV2_NODE_ID, ramp_torque2_nm);
        
        static uint32_t dbg_tx_count = 0;
        if (dbg_tx_count < 5) {
            Serial.printf("TX torque: %.3f, %.3f, %.3f Nm\n", ramp_torque0_nm, ramp_torque1_nm, ramp_torque2_nm);
            dbg_tx_count++;
        }
        
        last_torque0_nm = ramp_torque0_nm;
        last_torque1_nm = ramp_torque1_nm;
        last_torque2_nm = ramp_torque2_nm;
    }
    
    // Request encoder estimates periodically
    static unsigned long last_enc_req_ms = 0;
    if (millis() - last_enc_req_ms >= 10) {
        last_enc_req_ms = millis();
        sendGetEncoderEstimates(ODRV0_NODE_ID);
        sendGetEncoderEstimates(ODRV1_NODE_ID);
        sendGetEncoderEstimates(ODRV2_NODE_ID);
    }
    
    // Update joint degrees from encoder counts
    cnt0 = readEncoderCount(enc0);
    cnt1 = readEncoderCount(enc1);
    cnt2 = readEncoderCount(enc2);
    joint_deg0 = (cnt0 / (float)ENCODER_CPR_X4) * 360.0f;
    joint_deg1 = (cnt1 / (float)ENCODER_CPR_X4) * 360.0f;
    joint_deg2 = (cnt2 / (float)ENCODER_CPR_X4) * 360.0f;
    
    if (has_pos0 || has_pos1 || has_pos2) {
        Serial.print("FB,");
        Serial.print(millis()); Serial.print(",");
        Serial.print(last_pos0, 6); Serial.print(",");
        Serial.print(last_pos1, 6); Serial.print(",");
        Serial.print(last_pos2, 6); Serial.print(",");
        Serial.print(joint_deg0, 4); Serial.print(",");
        Serial.print(joint_deg1, 4); Serial.print(",");
        Serial.print(joint_deg2, 4); Serial.print(",");
        Serial.println((joint_deg0 + joint_deg1 + joint_deg2) / 3.0f, 4);
        last_fb_ms = millis();
        has_pos0 = has_pos1 = has_pos2 = false;
    }
    
    if (millis() - last_fb_ms > 2000) {
        Serial.println("WARN: feedback timeout >2s");
        last_fb_ms = millis();
    }
    
    // State broadcast @ 50 Hz
    static unsigned long last_state_ms = 0;
    if (millis() - last_state_ms >= 20) {
        last_state_ms = millis();
        
        unsigned long now_ms = millis();
        float dt_s = (now_ms - last_vel_calc_ms) / 1000.0f;
        if (dt_s > 0.001f) {
            current_vel_deg_s[0] = (joint_deg0 - last_joint_deg[0]) / dt_s;
            current_vel_deg_s[1] = (joint_deg1 - last_joint_deg[1]) / dt_s;
            current_vel_deg_s[2] = (joint_deg2 - last_joint_deg[2]) / dt_s;
            last_joint_deg[0] = joint_deg0;
            last_joint_deg[1] = joint_deg1;
            last_joint_deg[2] = joint_deg2;
            last_vel_calc_ms = now_ms;
        }
        
        float tau0_out = isnan(ramp_torque0_nm) ? 0.0f : ramp_torque0_nm;
        float tau1_out = isnan(ramp_torque1_nm) ? 0.0f : ramp_torque1_nm;
        float tau2_out = isnan(ramp_torque2_nm) ? 0.0f : ramp_torque2_nm;
        
        Serial.print("FB,");
        Serial.print(joint_deg0, 3); Serial.print(",");
        Serial.print(joint_deg1, 3); Serial.print(",");
        Serial.print(joint_deg2, 3); Serial.print(",");
        Serial.print(current_vel_deg_s[0], 3); Serial.print(",");
        Serial.print(current_vel_deg_s[1], 3); Serial.print(",");
        Serial.print(current_vel_deg_s[2], 3); Serial.print(",");
        Serial.print(tau0_out, 4); Serial.print(",");
        Serial.print(tau1_out, 4); Serial.print(",");
        Serial.print(tau2_out, 4); Serial.print(",");
        Serial.println(g_ctrl.motion_active ? 1 : 0);
    }
}

// ============================================================================
// Serial Command Processing (minimal - add your commands here)
// ============================================================================
void processSerialCommand() {
    if (!Serial.available()) return;
    
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) return;
    
    if (cmd == "STOP") {
        onBoardCtlHoldCurrent();
        setClosedLoop(false);
        ramp_torque0_nm = ramp_torque1_nm = ramp_torque2_nm = 0.0f;
        sendInputTorque(ODRV0_NODE_ID, 0.0f);
        sendInputTorque(ODRV1_NODE_ID, 0.0f);
        sendInputTorque(ODRV2_NODE_ID, 0.0f);
        Serial.println("INFO: STOPPED");
    }
    else if (cmd.startsWith("TORQUE ")) {
        float t0, t1, t2;
        if (sscanf(cmd.c_str(), "TORQUE %f %f %f", &t0, &t1, &t2) == 3) {
            ramp_torque0_nm = constrain(t0, -TORQUE_MAX_NM, TORQUE_MAX_NM);
            ramp_torque1_nm = constrain(t1, -TORQUE_MAX_NM, TORQUE_MAX_NM);
            ramp_torque2_nm = constrain(t2, -TORQUE_MAX_NM, TORQUE_MAX_NM);
            Serial.printf("TORQUE set: %.3f, %.3f, %.3f Nm\n", t0, t1, t2);
        }
    }
    else if (cmd.startsWith("GOTO ")) {
        float q0, q1, q2;
        if (sscanf(cmd.c_str(), "GOTO %f %f %f", &q0, &q1, &q2) == 3) {
            onBoardCtlSetTargetDeg(q0, q1, q2);
        }
    }
    else if (cmd == "PING") {
        // Test CAN ping to all nodes
        Serial.println("PING: sending...");
        sendGetEncoderEstimates(0);
        delay(5);
        sendGetEncoderEstimates(1);
        delay(5);
        sendGetEncoderEstimates(2);
        delay(50);
        Serial.printf("PONG: HB=[%d,%d,%d]\n",
            odrv_data[0].received_heartbeat,
            odrv_data[1].received_heartbeat,
            odrv_data[2].received_heartbeat);
    }
    else if (cmd == "ODRV") {
        // Request ODrive axis state via CAN
        Serial.println("Requesting ODrive states...");
        // ODrive cmd 0x07 = get_axis_state
        uint8_t data[1] = {0};
        sendCanMessage((0x07 << 5) | 1, data, 1);  // node 1
        delay(100);
        Serial.printf("node1 state=%d\n", odrv_data[1].last_heartbeat.Axis_State);
    }
    else if (cmd == "CLOSE" || cmd == "SETCL") {
        Serial.println("Setting closed loop for all...");
        clearBothErrors();
        setTorqueMode();
        delay(100);
        setClosedLoop(true);
        Serial.println("Done. Check ODRV for state.");
    }
    else if (cmd == "IDLE") {
        onBoardCtlHoldCurrent();
        setClosedLoop(false);
        ramp_torque0_nm = ramp_torque1_nm = ramp_torque2_nm = 0.0f;
        sendInputTorque(ODRV0_NODE_ID, 0.0f);
        sendInputTorque(ODRV1_NODE_ID, 0.0f);
        sendInputTorque(ODRV2_NODE_ID, 0.0f);
        Serial.println("INFO: IDLE mode");
    }
    else if (cmd == "STATUS") {
        Serial.printf("STATUS: HB=[%d,%d,%d] motion=%d tau=(%.3f,%.3f,%.3f) can_tx=%lu fail=%lu\n",
            odrv_data[0].received_heartbeat,
            odrv_data[1].received_heartbeat,
            odrv_data[2].received_heartbeat,
            g_ctrl.motion_active,
            ramp_torque0_nm, ramp_torque1_nm, ramp_torque2_nm,
            can_tx_count, can_tx_fail);
    }
    else if (cmd == "HELP") {
        Serial.println("Commands: STOP, TORQUE <t0> <t1> <t2>, GOTO <q0> <q1> <q2>, STATUS, ENC");
    }
    else if (cmd == "ENC") {
        Serial.printf("ENC: %d, %d, %d | %.2f, %.2f, %.2f deg\n",
            cnt0, cnt1, cnt2, joint_deg0, joint_deg1, joint_deg2);
    }
    else if (cmd.startsWith("LOCK ")) {
        int l0, l1, l2;
        if (sscanf(cmd.c_str(), "LOCK %d %d %d", &l0, &l1, &l2) == 3) {
            onBoardCtlSetLockedAxes(l0 != 0, l1 != 0, l2 != 0);
        }
    }
}
