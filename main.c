/**
 * ODrive TWAI Controller - ESP-IDF Version
 * Controls 3 ODrive motors via CAN bus using native ESP-IDF TWAI driver
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_system.h"
#include "driver/twai.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_intr_alloc.h"

#include "ODriveCAN.h"
#include "on_board_ctl.h"

static const char* TAG = "ODRIVE_CTL";

// ============================================================================
// Global State
// ============================================================================

CtrlState g_ctrl;
float g_joint_pos_deg[3] = {0.0f, 0.0f, 0.0f};
float g_joint_vel_deg_s[3] = {0.0f, 0.0f, 0.0f};

#define CAN_TX_PIN GPIO_NUM_1
#define CAN_RX_PIN GPIO_NUM_2

#define ENC0_PIN_A GPIO_NUM_4
#define ENC0_PIN_B GPIO_NUM_5
#define ENC0_PIN_Z GPIO_NUM_6
#define ENC1_PIN_A GPIO_NUM_7
#define ENC1_PIN_B GPIO_NUM_8
#define ENC1_PIN_Z GPIO_NUM_9
#define ENC2_PIN_A GPIO_NUM_10
#define ENC2_PIN_B GPIO_NUM_11
#define ENC2_PIN_Z GPIO_NUM_12

#define ODRV0_NODE_ID 0
#define ODRV1_NODE_ID 1
#define ODRV2_NODE_ID 2

#define TORQUE_TX_INTERVAL_MS 10
#define MAX_TORQUE_NM 150.0f
#define ENCODER_CPR 16384

// ============================================================================
// Global State
// ============================================================================

static bool twai_initialized = false;
static bool bus_off_recovering = false;
static bool closed_loop_requested = false;
static bool motion_active = false;
static bool locked_axes[3] = {false, false, false};

static ODriveUserData odrv_data[3];
static bool hb_received[3] = {false, false, false};

static uint32_t can_tx_count = 0;
static uint32_t can_tx_fail = 0;
static uint32_t can_rx_count = 0;

static float ramp_torque[3] = {0.0f, 0.0f, 0.0f};
static float last_torque[3] = {0.0f, 0.0f, 0.0f};

static volatile int32_t encoder_count[3] = {0, 0, 0};

static float last_pos[3] = {0.0f, 0.0f, 0.0f};
static bool has_pos[3] = {false, false, false};
static float joint_deg[3] = {0.0f, 0.0f, 0.0f};
static float vel_deg_s[3] = {0.0f, 0.0f, 0.0f};
static float last_joint_deg[3] = {0.0f, 0.0f, 0.0f};

// ============================================================================
// Forward Declarations
// ============================================================================

static void clear_errors(void);
static void set_torque_mode(void);
static void set_torque_ramp_rate(float rate_nm_s);
static void set_closed_loop(bool enable);
static void send_get_encoder_estimates(uint8_t node_id);

// ============================================================================
// UART
// ============================================================================

static QueueHandle_t uart_queue;

static void uart_init(void) {
    // Use GPIO43/44 for UART0 to avoid conflict with CAN (GPIO1/2)
    uart_config_t uart_config = {
        .baud_rate = 921600,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_0, &uart_config));
    // TX=GPIO43, RX=GPIO44 (USB-UART on most ESP32-S3 DevKit boards)
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM_0, 43, 44, 
                                  UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 16384, 8192, 20, &uart_queue, 0));
}

static void uart_write_str(const char* str) {
    uart_write_bytes(UART_NUM_0, str, strlen(str));
}

static void uart_write_fmt(const char* fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (len > 0) {
        uart_write_bytes(UART_NUM_0, buf, len);
    }
}

// ============================================================================
// Command Processing
// ============================================================================

static void process_command(const char* cmd) {
    if (strcmp(cmd, "STOP") == 0) {
        motion_active = false;
        ramp_torque[0] = ramp_torque[1] = ramp_torque[2] = 0.0f;
        set_closed_loop(false);
        uart_write_str("INFO: STOPPED\r\n");
    }
    else if (strncmp(cmd, "TORQUE ", 7) == 0) {
        float t0, t1, t2;
        if (sscanf(cmd + 7, "%f %f %f", &t0, &t1, &t2) == 3) {
            ramp_torque[0] = t0;
            ramp_torque[1] = t1;
            ramp_torque[2] = t2;
            uart_write_fmt("TORQUE set: %.3f, %.3f, %.3f Nm\r\n", t0, t1, t2);
        }
    }
    else if (cmd[0] == 'T' && cmd[2] == ':') {
        // Parse T0:, T1:, T2: format from Python bridge
        int axis = cmd[1] - '0';
        float torque = atof(cmd + 3);
        if (axis >= 0 && axis < 3) {
            ramp_torque[axis] = torque;
        }
    }
    else if (strcmp(cmd, "CLOSE") == 0) {
        clear_errors();
        vTaskDelay(pdMS_TO_TICKS(50));
        set_torque_mode();
        vTaskDelay(pdMS_TO_TICKS(50));
        set_closed_loop(true);
        vTaskDelay(pdMS_TO_TICKS(200));
        bool ok = true;
        for (int i = 0; i < 3; i++) {
            if (odrv_data[i].last_heartbeat.axis_state != AXIS_STATE_CLOSED_LOOP_CONTROL) {
                ok = false;
                uart_write_fmt("WARN: node%d not in CLOSED_LOOP (state=%d)\r\n",
                    i, odrv_data[i].last_heartbeat.axis_state);
            }
        }
        if (ok) uart_write_str("INFO: Closed loop enabled\r\n");
    }
    else if (strcmp(cmd, "IDLE") == 0) {
        motion_active = false;
        ramp_torque[0] = ramp_torque[1] = ramp_torque[2] = 0.0f;
        set_closed_loop(false);
        uart_write_str("INFO: IDLE mode\r\n");
    }
    else if (strcmp(cmd, "STATUS") == 0) {
        uart_write_fmt("STATUS: HB=[%d,%d,%d] motion=%d\r\n",
            hb_received[0], hb_received[1], hb_received[2],
            g_ctrl.motion_active ? 1 : 0);
        for (int i = 0; i < 3; i++) {
            uart_write_fmt("  node%d: state=%d req=%d\r\n",
                i, odrv_data[i].last_heartbeat.axis_state,
                odrv_data[i].last_heartbeat.requested_state);
        }
    }
    else if (strcmp(cmd, "PING") == 0) {
        uart_write_str("PING: sending...\r\n");
        for (int i = 0; i < 3; i++) {
            send_get_encoder_estimates(i);
            vTaskDelay(pdMS_TO_TICKS(5));
        }
        vTaskDelay(pdMS_TO_TICKS(50));
        uart_write_fmt("PONG: HB=[%d,%d,%d]\r\n",
            hb_received[0], hb_received[1], hb_received[2]);
    }
    else if (strcmp(cmd, "ENC") == 0) {
        uart_write_fmt("ENC: %d, %d, %d | %.2f, %.2f, %.2f deg\r\n",
            encoder_count[0], encoder_count[1], encoder_count[2],
            joint_deg[0], joint_deg[1], joint_deg[2]);
    }
    else if (strcmp(cmd, "CLEAR") == 0) {
        clear_errors();
        uart_write_str("INFO: Errors cleared\r\n");
    }
    else if (strcmp(cmd, "HELP") == 0) {
        uart_write_str("Commands:\r\n");
        uart_write_str("  STOP, TORQUE <t0> <t1> <t2>\r\n");
        uart_write_str("  GOTO <q0> <q1> <q2> - Quintic trajectory\r\n");
        uart_write_str("  HOME - set current as zero\r\n");
        uart_write_str("  CLOSE, IDLE, CLEAR, STATUS, PING\r\n");
        uart_write_str("  RAMP <rate> - set torque ramp rate (Nm/s)\r\n");
    }
    else if (strncmp(cmd, "RAMP ", 5) == 0) {
        float rate;
        if (sscanf(cmd + 5, "%f", &rate) == 1) {
            set_torque_ramp_rate(rate);
            uart_write_fmt("RAMP: torque ramp rate = %.1f Nm/s\r\n", rate);
        }
    }
    else if (strncmp(cmd, "GOTO ", 5) == 0) {
        float q0, q1, q2;
        if (sscanf(cmd + 5, "%f %f %f", &q0, &q1, &q2) == 3) {
            onBoardCtlSetTargetDeg(q0, q1, q2);
            uart_write_fmt("GOTO: (%.2f, %.2f, %.2f) deg\r\n", q0, q1, q2);
        }
    }
    else if (strcmp(cmd, "HOME") == 0) {
        onBoardCtlHoldCurrent();
        uart_write_str("INFO: HOME set, torque zeroed\r\n");
    }
    else if (strlen(cmd) > 0) {
        uart_write_fmt("WARN: unknown cmd: %s\r\n", cmd);
    }
}

static void uart_task(void* arg) {
    char cmd_buf[256];
    int buf_idx = 0;
    
    while (1) {
        uint8_t ch;
        if (uart_read_bytes(UART_NUM_0, &ch, 1, portMAX_DELAY) == 1) {
            if (ch == '\r' || ch == '\n') {
                cmd_buf[buf_idx] = '\0';
                if (buf_idx > 0) {
                    process_command(cmd_buf);
                    buf_idx = 0;
                }
            } else if (buf_idx < sizeof(cmd_buf) - 1) {
                cmd_buf[buf_idx++] = ch;
            }
        }
    }
}

// ============================================================================
// Encoder (GPIO ISR - Quadrature x4 Decoding)
// ============================================================================

typedef struct {
    volatile int32_t count;
    uint8_t last_AB;
    uint8_t pinA;
    uint8_t pinB;
    int direction;
} encoder_state_t;

static encoder_state_t encoder_states[3];

void IRAM_ATTR encoder_isr(void* arg) {
    encoder_state_t* enc = (encoder_state_t*)arg;
    uint8_t a = gpio_get_level((gpio_num_t)enc->pinA) ? 1 : 0;
    uint8_t b = gpio_get_level((gpio_num_t)enc->pinB) ? 1 : 0;
    uint8_t AB = (a << 0) | (b << 1);
    
    switch ((enc->last_AB << 2) | AB) {
        case 0b0001: case 0b0111: case 0b1110: case 0b1000: enc->count++; break;
        case 0b0010: case 0b1011: case 0b1101: case 0b0100: enc->count--; break;
    }
    enc->last_AB = AB;
}

static void encoder_init(void) {
    static const gpio_num_t enc_a_pins[3] = {ENC0_PIN_A, ENC1_PIN_A, ENC2_PIN_A};
    static const gpio_num_t enc_b_pins[3] = {ENC0_PIN_B, ENC1_PIN_B, ENC2_PIN_B};
    static const int enc_dirs[3] = {1, 1, 1}; // direction per axis
    
    // Install ISR service
    gpio_install_isr_service(ESP_INTR_FLAG_IRAM);
    
    for (int i = 0; i < 3; i++) {
        encoder_states[i].pinA = (uint8_t)enc_a_pins[i];
        encoder_states[i].pinB = (uint8_t)enc_b_pins[i];
        encoder_states[i].direction = enc_dirs[i];
        encoder_states[i].count = 0;
        encoder_states[i].last_AB = 
            (gpio_get_level(enc_a_pins[i]) ? 1 : 0) | 
            (gpio_get_level(enc_b_pins[i]) ? 2 : 0);
        
        gpio_config_t io_conf = {
            .pin_bit_mask = (1ULL << enc_a_pins[i]) | (1ULL << enc_b_pins[i]),
            .mode = GPIO_MODE_INPUT,
            .pull_up_en = GPIO_PULLUP_ENABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_ANYEDGE,
        };
        gpio_config(&io_conf);
        gpio_isr_handler_add(enc_a_pins[i], encoder_isr, &encoder_states[i]);
        gpio_isr_handler_add(enc_b_pins[i], encoder_isr, &encoder_states[i]);
    }
}

static float get_joint_deg(uint8_t enc_id) {
    int32_t count = encoder_states[enc_id].count * encoder_states[enc_id].direction;
    return ((float)count / (float)ENCODER_CPR) * 360.0f;
}

// ============================================================================
// TWAI (CAN)
// ============================================================================

static bool twai_init(void) {
    twai_general_config_t g_config = {
        .mode = TWAI_MODE_NORMAL,
        .tx_io = CAN_TX_PIN,
        .rx_io = CAN_RX_PIN,
        .clkout_io = GPIO_NUM_NC,
        .bus_off_io = GPIO_NUM_NC,
        .tx_queue_len = 64,
        .rx_queue_len = 256,
        .alerts_enabled = TWAI_ALERT_ALL,
        .clkout_divider = 0,
    };
    
    twai_timing_config_t t_config = TWAI_TIMING_CONFIG_250KBITS();
    
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();
    
    esp_err_t err = twai_driver_install(&g_config, &t_config, &f_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "TWAI install failed: %s", esp_err_to_name(err));
        return false;
    }
    
    err = twai_start();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "TWAI start failed: %s", esp_err_to_name(err));
        twai_driver_uninstall();
        return false;
    }
    
    return true;
}

static bool send_can_message(uint32_t can_id, const uint8_t* data, uint8_t dlc) {
    if (!twai_initialized) return false;
    
    twai_message_t msg = {
        .identifier = can_id,
        .flags = TWAI_MSG_FLAG_NONE,
        .data_length_code = dlc,
    };
    
    for (int i = 0; i < dlc && i < 8; i++) {
        msg.data[i] = data[i];
    }
    
    esp_err_t err = twai_transmit(&msg, pdMS_TO_TICKS(10));
    
    if (err == ESP_OK) {
        can_tx_count++;
        return true;
    }
    
    can_tx_fail++;
    return false;
}

static void send_axis_state(uint8_t node_id, uint32_t axis_state) {
    uint8_t data[4];
    write_i32_le(data, (int32_t)axis_state);
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_SET_AXIS_REQUESTED_STATE;
    uart_write_fmt("[DBG] TX SET_STATE node=%d state=%d can_id=0x%03X\r\n", node_id, axis_state, can_id);
    send_can_message(can_id, data, 4);
}

static void send_input_torque(uint8_t node_id, float torque_nm) {
    uint8_t data[4];
    // Convert joint torque to motor torque: tau_motor = tau_joint / (gear_ratio * efficiency)
    float tau_motor = jointToMotorTorque(node_id, torque_nm);
    write_float_le(data, tau_motor);
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_SET_INPUT_TORQUE;
    send_can_message(can_id, data, 4);
}

static void send_torque_ramp_rate(uint8_t node_id, float ramp_rate_nm_s) {
    uint8_t data[4];
    write_float_le(data, ramp_rate_nm_s);
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_SET_INPUT_TORQUE_RAMP_RATE;
    send_can_message(can_id, data, 4);
}

static void send_controller_mode(uint8_t node_id, int32_t control_mode, int32_t input_mode) {
    uint8_t data[8];
    write_i32_le(&data[0], control_mode);
    write_i32_le(&data[4], input_mode);
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_SET_CONTROLLER_MODES;
    send_can_message(can_id, data, 8);
}

static void send_get_encoder_estimates(uint8_t node_id) {
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_GET_ENCODER_ESTIMATES;
    send_can_message(can_id, NULL, 0);
}

static void send_clear_errors(uint8_t node_id) {
    uint32_t can_id = (node_id << 5) | ODRIVE_CMD_CLEAR_ERRORS;
    send_can_message(can_id, NULL, 0);
}

static void send_save_config(uint8_t node_id) {
    uint32_t can_id = (node_id << 5) | 0x16;
    send_can_message(can_id, NULL, 0);
}

static void clear_errors(void) {
    send_clear_errors(ODRV0_NODE_ID);
    send_clear_errors(ODRV1_NODE_ID);
    send_clear_errors(ODRV2_NODE_ID);
}

// Default ramp rate: 10 Nm/s (adjustable via RAMP command)
static float s_torque_ramp_rate = 10.0f;

static void set_torque_mode(void) {
    // INPUT_MODE_TORQUE_RAMP: torque được ramp với tốc độ có thể cấu hình
    send_controller_mode(ODRV0_NODE_ID, CONTROL_MODE_TORQUE_CONTROL, INPUT_MODE_TORQUE_RAMP);
    vTaskDelay(pdMS_TO_TICKS(5));
    send_controller_mode(ODRV1_NODE_ID, CONTROL_MODE_TORQUE_CONTROL, INPUT_MODE_TORQUE_RAMP);
    vTaskDelay(pdMS_TO_TICKS(5));
    send_controller_mode(ODRV2_NODE_ID, CONTROL_MODE_TORQUE_CONTROL, INPUT_MODE_TORQUE_RAMP);
    vTaskDelay(pdMS_TO_TICKS(10));
    
    // Set ramp rate for all axes
    send_torque_ramp_rate(ODRV0_NODE_ID, s_torque_ramp_rate);
    send_torque_ramp_rate(ODRV1_NODE_ID, s_torque_ramp_rate);
    send_torque_ramp_rate(ODRV2_NODE_ID, s_torque_ramp_rate);
    
    ESP_LOGI(TAG, "Torque mode: CONTROL_MODE_TORQUE + INPUT_MODE_TORQUE_RAMP (%.1f Nm/s)", s_torque_ramp_rate);
}

static void set_torque_ramp_rate(float rate_nm_s) {
    s_torque_ramp_rate = rate_nm_s;
    send_torque_ramp_rate(ODRV0_NODE_ID, rate_nm_s);
    send_torque_ramp_rate(ODRV1_NODE_ID, rate_nm_s);
    send_torque_ramp_rate(ODRV2_NODE_ID, rate_nm_s);
    ESP_LOGI(TAG, "Torque ramp rate set to %.1f Nm/s", rate_nm_s);
}

static void set_closed_loop(bool enable) {
    uint32_t target = enable ? AXIS_STATE_CLOSED_LOOP_CONTROL : AXIS_STATE_IDLE;
    send_axis_state(ODRV0_NODE_ID, target);
    send_axis_state(ODRV1_NODE_ID, target);
    send_axis_state(ODRV2_NODE_ID, target);
    closed_loop_requested = enable;
}

// ============================================================================
// CAN RX Processing
// ============================================================================

static void process_can_rx(void);

static bool verify_closed_loop(uint32_t timeout_ms) {
    int64_t t0 = esp_timer_get_time() / 1000;
    bool logged_once = false;
    
    while ((esp_timer_get_time() / 1000 - t0) < timeout_ms) {
        // Process incoming CAN messages to update heartbeat states
        process_can_rx();
        
        bool all_closed = true;
        for (int i = 0; i < 3; i++) {
            uint16_t state = odrv_data[i].last_heartbeat.axis_state;
            if (state != AXIS_STATE_CLOSED_LOOP_CONTROL) {
                all_closed = false;
                if (!logged_once) {
                    uart_write_fmt("WAIT: node%d state=%d (expecting %d)\r\n",
                        i, state, AXIS_STATE_CLOSED_LOOP_CONTROL);
                }
            }
        }
        if (all_closed) return true;
        
        logged_once = true;
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    return false;
}

static void process_can_rx(void) {
    twai_message_t rx_msg;
    
    // Drain ALL pending CAN messages (like Arduino pumpEvents)
    while (twai_receive(&rx_msg, pdMS_TO_TICKS(0)) == ESP_OK) {
        can_rx_count++;
        uint8_t node_id = (rx_msg.identifier >> 5) & 0x1F;
        uint8_t cmd = rx_msg.identifier & 0x1F;
        
        if (node_id < 3) {
            switch (cmd) {
                case ODRIVE_CMD_HEARTBEAT: {
                    if (rx_msg.data_length_code >= 7) {
                        odrv_data[node_id].last_heartbeat.axis_state = 
                            rx_msg.data[4] | (rx_msg.data[5] << 8);
                        odrv_data[node_id].last_heartbeat.requested_state = rx_msg.data[6];
                        odrv_data[node_id].received_heartbeat = true;
                        hb_received[node_id] = true;
                        // Debug: print raw heartbeat bytes
                        uart_write_fmt("[DBG] HB node%d: err=0x%02X mtr=0x%02X enc=0x%02X sens=0x%02X state=%d req=%d\r\n",
                            node_id, rx_msg.data[0], rx_msg.data[1], rx_msg.data[2], rx_msg.data[3],
                            odrv_data[node_id].last_heartbeat.axis_state,
                            odrv_data[node_id].last_heartbeat.requested_state);
                    }
                    break;
                }
                case ODRIVE_CMD_GET_ENCODER_ESTIMATES: {
                    if (rx_msg.data_length_code >= 8) {
                        float pos_est, vel_est;
                        memcpy(&pos_est, rx_msg.data, sizeof(float));
                        memcpy(&vel_est, rx_msg.data + 4, sizeof(float));
                        odrv_data[node_id].last_feedback.Pos_Estimate = pos_est;
                        odrv_data[node_id].last_feedback.Vel_Estimate = vel_est;
                        odrv_data[node_id].received_feedback = true;
                        last_pos[node_id] = pos_est;
                        has_pos[node_id] = true;
                    }
                    break;
                }
            }
        }
    }
}

static void check_can_health(void) {
    uint32_t alerts = 0;
    twai_read_alerts(&alerts, pdMS_TO_TICKS(0));
    
    if (alerts & TWAI_ALERT_BUS_OFF) {
        if (!bus_off_recovering) {
            bus_off_recovering = true;
            twai_initiate_recovery();
            vTaskDelay(pdMS_TO_TICKS(250));
        }
    }
    if (alerts & TWAI_ALERT_BUS_RECOVERED) {
        twai_start();
        bus_off_recovering = false;
    }
    if (alerts & TWAI_ALERT_TX_FAILED) {
        can_tx_fail++;
    }
}

static bool wait_heartbeat(uint8_t node_id, const char* label, uint32_t timeout_ms) {
    int64_t t0 = esp_timer_get_time() / 1000;
    odrv_data[node_id].received_heartbeat = false;
    
    while (!odrv_data[node_id].received_heartbeat) {
        process_can_rx();
        if ((esp_timer_get_time() / 1000 - t0) > timeout_ms) {
            return false;
        }
        vTaskDelay(pdMS_TO_TICKS(2));
    }
    return true;
}

// ============================================================================
// Control Task
// ============================================================================

static void control_task(void* arg) {
    int64_t last_enc_req_us = 0;
    int64_t last_vel_us = 0;
    int64_t last_debug_us = 0;
    int64_t last_can_tx_us = 0;
    
    // 100Hz torque control, 100Hz encoder request (like Arduino)
    const int64_t TORQUE_INTERVAL_US = 10000;   // 10ms = 100Hz
    const int64_t ENC_INTERVAL_US = 10000;       // 10ms = 100Hz
    
    while (1) {
        int64_t now_us = esp_timer_get_time();
        
        process_can_rx();
        check_can_health();
        
        // Encoder request at 50Hz (reduce CAN load)
        if (now_us - last_enc_req_us >= ENC_INTERVAL_US) {
            last_enc_req_us = now_us;
            send_get_encoder_estimates(ODRV0_NODE_ID);
            send_get_encoder_estimates(ODRV1_NODE_ID);
            send_get_encoder_estimates(ODRV2_NODE_ID);
        }
        
        // Update CTC control at 100Hz
        onBoardCtlUpdate();
        
        // Apply CTC torques - throttle CAN TX to reduce bus load
        if (now_us - last_can_tx_us >= TORQUE_INTERVAL_US) {
            last_can_tx_us = now_us;
            
            float tau[3];
            for (int i = 0; i < 3; i++) {
                if (g_ctrl.locked_axes[i]) {
                    tau[i] = 0.0f;
                } else if (g_ctrl.motion_active) {
                    tau[i] = g_ctrl.tau_joint[i];
                } else {
                    tau[i] = ramp_torque[i];
                }
                tau[i] = (tau[i] < -MAX_TORQUE_NM) ? -MAX_TORQUE_NM : (tau[i] > MAX_TORQUE_NM) ? MAX_TORQUE_NM : tau[i];
            }
            
            // Send torque to all 3 ODrives
            send_input_torque(ODRV0_NODE_ID, tau[0]);
            send_input_torque(ODRV1_NODE_ID, tau[1]);
            send_input_torque(ODRV2_NODE_ID, tau[2]);
            
            last_torque[0] = tau[0];
            last_torque[1] = tau[1];
            last_torque[2] = tau[2];
        }
        
        for (int i = 0; i < 3; i++) {
            joint_deg[i] = get_joint_deg(i);
            g_joint_pos_deg[i] = joint_deg[i];
            g_joint_vel_deg_s[i] = vel_deg_s[i];
        }
        
        if (now_us - last_vel_us >= 20000) {
            float dt_s = (now_us - last_vel_us) / 1000000.0f;
            if (dt_s > 0.001f) {
                for (int i = 0; i < 3; i++) {
                    vel_deg_s[i] = (joint_deg[i] - last_joint_deg[i]) / dt_s;
                }
            }
            last_vel_us = now_us;
            for (int i = 0; i < 3; i++) {
                last_joint_deg[i] = joint_deg[i];
            }
        }
        
        // Debug TX status every 5 seconds (reduce serial traffic)
        if (now_us - last_debug_us >= 5000000) {
            last_debug_us = now_us;
            // uart_write_fmt("TX: can_tx=%lu fail=%lu ...\r\n", ...);  // Commented out to reduce traffic
        }
        
        // Only print when all 3 axes have received feedback (like Arduino)
        // Send immediately without throttle - let PC handle timing
        if (has_pos[0] && has_pos[1] && has_pos[2]) {
            uart_write_fmt("FB,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%d\r\n",
                last_pos[0], last_pos[1], last_pos[2],
                joint_deg[0], joint_deg[1], joint_deg[2],
                last_torque[0], last_torque[1], last_torque[2],
                g_ctrl.motion_active ? 1 : 0);
            has_pos[0] = has_pos[1] = has_pos[2] = false;
        }
        
        // No vTaskDelay - loop runs as fast as possible like Arduino
    }
}

// ============================================================================
// Main
// ============================================================================

void app_main(void)
{
    uart_init();
    
    encoder_init();
    onBoardCtlInit();
    
    if (!twai_init()) {
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    twai_initialized = true;
    
    vTaskDelay(pdMS_TO_TICKS(100));
    
    wait_heartbeat(ODRV0_NODE_ID, "node0", 5000);
    wait_heartbeat(ODRV1_NODE_ID, "node1", 5000);
    wait_heartbeat(ODRV2_NODE_ID, "node2", 5000);

    clear_errors();
    vTaskDelay(pdMS_TO_TICKS(100));

    set_torque_mode();
    vTaskDelay(pdMS_TO_TICKS(100));

    set_closed_loop(true);

    if (!verify_closed_loop(3000)) {
        uart_write_str("WARN: not all nodes reached CLOSED_LOOP\r\n");
        // Print which nodes failed
        for (int i = 0; i < 3; i++) {
            uart_write_fmt("  node%d: axis_state=%d\r\n",
                i, odrv_data[i].last_heartbeat.axis_state);
        }
    } else {
        uart_write_str("INFO: all nodes in CLOSED_LOOP\r\n");
    }
    
    xTaskCreatePinnedToCore(control_task, "control", 8192, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(uart_task, "uart", 16384, NULL, 5, NULL, 1);
    
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
