/**
 * ESP32-S3 ODrive CAN Bridge
 * ===============================
 * Simple pass-through bridge: PC <-> Serial <-> ESP32 <-> CAN <-> ODrive
 * All CTC computation on PC
 * ESP32 handles: CAN communication, Quadrature encoder reading (A/B/Z)
 * 
 * Protocol:
 *   From PC: TORQUE, POS, STATE, MODE, CLEAR, CLOSE, IDLE, HOME, ENC, STATUS, PING
 *   To PC:   HB, FB, JPOS, ENC reply
 */

 #include <stdio.h>
 #include <string.h>
 #include <stdlib.h>
 #include <stdarg.h>
 #include <math.h>
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
 
 // ============================================================================
 // Logging
 // ============================================================================
 static const char* TAG = "BRIDGE";
 
 // ============================================================================
 // GPIO Pin Configuration
 // ============================================================================
 // CAN pins
 #define CAN_TX_PIN GPIO_NUM_1
 #define CAN_RX_PIN GPIO_NUM_2
 
// Encoder pins (A/B/Z for 3 axes)
#define ENC0_PIN_A GPIO_NUM_4
#define ENC0_PIN_B GPIO_NUM_5
#define ENC0_PIN_Z GPIO_NUM_6
#define ENC1_PIN_A GPIO_NUM_7
#define ENC1_PIN_B GPIO_NUM_8
#define ENC1_PIN_Z GPIO_NUM_9
#define ENC2_PIN_A GPIO_NUM_10
#define ENC2_PIN_B GPIO_NUM_11
#define ENC2_PIN_Z GPIO_NUM_12

// Limit switch pins (knee joint = axis 1)
// EXT (duỗi thẳng) = home position (q_feedback = 0°)
// FLEX (gập hết)   = q_max (q_feedback = -q_max)
// Active LOW (pullup nội, chưa bấm = HIGH, bấm = LOW).
#define LIMIT_KNEE_EXT_GPIO  GPIO_NUM_41
#define LIMIT_KNEE_FLEX_GPIO GPIO_NUM_42

// Encoder CPR (x4 = 16384 counts/rev for 4096 CPR encoder)
#define ENCODER_CPR_X4 16384

// Homing config (knee joint = axis 1 only — hip/toe không có switch)
#define HOME_KNEE_AXIS        1
// Seek velocity (motor side, rev/s). 0.05 rev/s = 18 deg/s → ~0.36°/s ở output (gear 50:1).
// Rất chậm, an toàn cho switch + tránh DC_BUS_OVER_REGEN_CURRENT khi brake.
#define HOME_SEEK_VEL_REV_S   0.05f
// vel_gain/vel_integrator_gain set trước khi SEEK (override default yếu).
// Default vel_gain=0.166 → motor stall dưới gravity. Tăng vừa đủ để mạnh, không overshoot.
// Quá cao → motor overshoot vel_target → brake mạnh → DC_BUS_OVER_REGEN_CURRENT.
#define HOME_SEEK_VEL_GAIN         0.5f   // Nm / (rev/s)
#define HOME_SEEK_VEL_INT_GAIN     1.0f   // Nm / rev
// Vel ramp rate (default ODrive = 1.0 rev/s/s). Càng nhỏ → ramp càng mượt.
#define HOME_SEEK_VEL_RAMP_RATE    1.0f   // rev/s/s → 0→0.05 rev/s mất 50ms → mượt
#define HOME_TIMEOUT_MS       30000  // 30s cho mỗi pha
 
 // ============================================================================
 // Constants
 // ============================================================================
 #define GEAR_RATIO_0 100.0f   // Hip
 #define GEAR_RATIO_1 50.0f    // Knee
 #define GEAR_RATIO_2 50.0f    // Ankle
 
 #define ODRV0_NODE_ID 0
 #define ODRV1_NODE_ID 1
 #define ODRV2_NODE_ID 2
 
// ============================================================================
// Bridge State
// ============================================================================
static bool twai_initialized = false;
static bool bus_off_recovering = false;
static uint32_t can_tx_count = 0;
static uint32_t can_tx_fail = 0;
static uint32_t can_rx_count = 0;

// CAN RX state
static bool hb_received[3] = {false, false, false};
static uint16_t axis_state[3] = {0, 0, 0};
static float motor_pos[3] = {0.0f, 0.0f, 0.0f};
static float motor_vel[3] = {0.0f, 0.0f, 0.0f};

// PC-side command mirror state
// Bridge mode: PC tính CTC, ESP32 chỉ forward torque/position xuống ODrive.
static float last_torque_nm[3] = {0.0f, 0.0f, 0.0f};
static float last_position_rev[3] = {0.0f, 0.0f, 0.0f};
static bool  motion_active = false;       // 1 khi đang chạy GOTO/motion
static bool  closed_loop_state = false;   // 1 sau CLOSE, 0 sau IDLE
static bool  locked_axes[3] = {false, false, false};

// ============================================================================
// Homing state machine — Knee joint (axis 1) dùng 1 limit switch (EXT)
// ----------------------------------------------------------------------------
// Flow:
//   IDLE → (pending) → SEEK_EXT (quay dir=+1 cùng chiều moment dương,
//                              chờ GPIO 41 pressed)
//       → DONE: brake, reset count=0 tại EXT, gửi "KOK,<phi_ext>,<q_max>"
//
// q_max_deg là do PC cung cấp qua command "HOME <q_max>".
//
// Sau calib (PC side):
//   q_feedback (knee) = phi_ext - encoder_raw    (deg, q_feedback=0 khi duỗi thẳng)
//                      = -q_max                  (khi gập hết)
//
// ============================================================================
typedef enum {
    HOME_IDLE = 0,
    HOME_KNEE_SEEK_EXT,   // tìm công tắc duỗi (GPIO 41)
    HOME_KNEE_DONE,
} home_state_t;

static volatile home_state_t home_state = HOME_IDLE;
static volatile int64_t      home_start_us = 0;
static volatile int32_t      home_phi_extended = 0;   // encoder count tại GPIO 41 (extended)
static volatile int          home_motor_dir    = +1;  // +1 = forward (toward EXT/duỗi, cùng chiều moment dương)
static volatile bool         home_pending      = false;
// q_max_deg nhận từ command "HOME <q_max>" (mặc định 90° nếu PC không gửi).
static volatile float        home_q_max_deg    = 90.0f;

// Limit switch state (volatile — đọc từ ISR + main loop)
static volatile bool         g_limit_ext_active = false;   // GPIO 41 = LOW (pressed)
static volatile int64_t      g_limit_ext_us = 0;

// Helper: đọc trạng thái switch (active = pressed)
static inline bool limit_ext_is_pressed(void) { return g_limit_ext_active; }
// Brake khi torque = 0 và closed_loop để tránh ODrive tự drift
static int64_t last_pc_cmd_us = 0;

// ============================================================================
// Velocity LP filter (ESP32-side, khớp công thức PC: alpha = 2π·fc / (2π·fc + fs))
// ----------------------------------------------------------------------------
// Lý do cần: raw vel = (joint_deg - last_joint_deg) / dt → 1 LSB sai ở encoder
// khi motor đứng yên tạo spike lớn (vd 1/0.01 ≈ 100 deg/s từ 1 count drift).
// LP ở ESP32 triệt spike trước khi gửi lên PC; PC vẫn giữ LP riêng để mượt thêm.
// Corner 50Hz (vs PC 80Hz) là tầng 1 nhẹ, chỉ anti-aliasing; PC làm phần mượt chính.
// ============================================================================
#define PI_F 6.2831853f
#define VEL_LP_FC_HZ 50.0f
#define VEL_LP_FS_HZ 100.0f
// alpha = 2π·fc / (2π·fc + fs) = 2π·50 / (2π·50 + 100) ≈ 0.758
// Dùng PI_F thay vì M_PI macro để không phụ thuộc _USE_MATH_DEFINES.
static const float vel_lp_alpha = (PI_F * VEL_LP_FC_HZ) /
                                  (PI_F * VEL_LP_FC_HZ + VEL_LP_FS_HZ);
static float vel_lp_prev[3] = {0.0f, 0.0f, 0.0f};
 
 // ============================================================================
 // Quadrature Encoder (x4 + Z-index)
 // ============================================================================
 typedef struct {
     volatile int32_t count;
     volatile bool z_pulse;
     volatile int32_t z_count;
     uint8_t last_AB;
     uint8_t pinA, pinB, pinZ;
     int direction;
 } encoder_state_t;
 
 static encoder_state_t encoder_states[3];
 static volatile bool g_z_index_detected[3] = {false, false, false};
 static int64_t g_last_z_us[3] = {0, 0, 0};
 
 void IRAM_ATTR encoder_isr(void* arg) {
     encoder_state_t* enc = (encoder_state_t*)arg;
     uint8_t a = gpio_get_level((gpio_num_t)enc->pinA) ? 1 : 0;
     uint8_t b = gpio_get_level((gpio_num_t)enc->pinB) ? 1 : 0;
     uint8_t z = gpio_get_level((gpio_num_t)enc->pinZ) ? 1 : 0;
     uint8_t AB = (a << 0) | (b << 1);
     
     switch ((enc->last_AB << 2) | AB) {
         case 0b0001: case 0b0111: case 0b1110: case 0b1000: enc->count++; break;
         case 0b0010: case 0b1011: case 0b1101: case 0b0100: enc->count--; break;
     }
     enc->last_AB = AB;
     
     if (z && !enc->z_pulse) {
         enc->z_pulse = true;
         enc->z_count = enc->count;
     } else if (!z) {
         enc->z_pulse = false;
     }
 }
 
void IRAM_ATTR z_index_isr(void* arg) {
    encoder_state_t* enc = (encoder_state_t*)arg;
    int enc_id = -1;
    for (int i = 0; i < 3; i++) {
        if (&encoder_states[i] == arg) { enc_id = i; break; }
    }
    if (gpio_get_level((gpio_num_t)enc->pinZ)) {
        enc->z_count = enc->count;
        g_z_index_detected[enc_id] = true;
        g_last_z_us[enc_id] = esp_timer_get_time();
    }
}

// ============================================================================
// Limit switch ISR (knee joint = axis 1)
// ----------------------------------------------------------------------------
// Active LOW: chưa bấm = HIGH (pullup nội), bấm = LOW.
// ISR set flag + lưu timestamp; home_task sẽ đọc flag ở main loop.
// ============================================================================
void IRAM_ATTR limit_ext_isr(void* arg) {
    bool pressed = (gpio_get_level(LIMIT_KNEE_EXT_GPIO) == 0);  // active LOW
    g_limit_ext_active = pressed;
    if (pressed) g_limit_ext_us = esp_timer_get_time();
}

// (Chỉ dùng 1 limit switch EXT cho homing — FLEX GPIO42 không cần nữa.)

static void encoder_init(void) {
     static const gpio_num_t a_pins[3] = {ENC0_PIN_A, ENC1_PIN_A, ENC2_PIN_A};
     static const gpio_num_t b_pins[3] = {ENC0_PIN_B, ENC1_PIN_B, ENC2_PIN_B};
     static const gpio_num_t z_pins[3] = {ENC0_PIN_Z, ENC1_PIN_Z, ENC2_PIN_Z};
     static const int dirs[3] = {1, 1, 1};
     
     gpio_install_isr_service(ESP_INTR_FLAG_IRAM);
     
     for (int i = 0; i < 3; i++) {
         encoder_states[i].pinA = (uint8_t)a_pins[i];
         encoder_states[i].pinB = (uint8_t)b_pins[i];
         encoder_states[i].pinZ = (uint8_t)z_pins[i];
         encoder_states[i].direction = dirs[i];
         encoder_states[i].count = 0;
         encoder_states[i].z_count = 0;
         encoder_states[i].z_pulse = false;
         encoder_states[i].last_AB = 
             (gpio_get_level(a_pins[i]) ? 1 : 0) | 
             (gpio_get_level(b_pins[i]) ? 2 : 0);
         
         gpio_config_t io_conf = {
             .pin_bit_mask = (1ULL << a_pins[i]) | (1ULL << b_pins[i]),
             .mode = GPIO_MODE_INPUT,
             .pull_up_en = GPIO_PULLUP_ENABLE,
             .pull_down_en = GPIO_PULLDOWN_DISABLE,
             .intr_type = GPIO_INTR_ANYEDGE,
         };
         gpio_config(&io_conf);
         gpio_isr_handler_add(a_pins[i], encoder_isr, &encoder_states[i]);
         gpio_isr_handler_add(b_pins[i], encoder_isr, &encoder_states[i]);
         
         gpio_config_t z_conf = {
             .pin_bit_mask = (1ULL << z_pins[i]),
             .mode = GPIO_MODE_INPUT,
             .pull_up_en = GPIO_PULLUP_ENABLE,
             .pull_down_en = GPIO_PULLDOWN_DISABLE,
             .intr_type = GPIO_INTR_POSEDGE,
         };
        gpio_config(&z_conf);
        gpio_isr_handler_add(z_pins[i], z_index_isr, &encoder_states[i]);
    }
    
    // ── Limit switch EXT cho knee (axis 1) ──
    // (Chỉ dùng 1 switch — FLEX GPIO42 bỏ qua.)
    gpio_config_t limit_ext_conf = {
        .pin_bit_mask = (1ULL << LIMIT_KNEE_EXT_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_ANYEDGE,
    };
    gpio_config(&limit_ext_conf);
    gpio_isr_handler_add(LIMIT_KNEE_EXT_GPIO, limit_ext_isr, NULL);
    g_limit_ext_active = (gpio_get_level(LIMIT_KNEE_EXT_GPIO) == 0);  // sync initial state
    
    ESP_LOGI(TAG, "Encoders: A/B/Z, %d CPR (x4)", ENCODER_CPR_X4);
    ESP_LOGI(TAG, "Knee limit switch EXT=GPIO%d (active LOW). q_max set via 'HOME <deg>'.",
             LIMIT_KNEE_EXT_GPIO);
}
 
 static float get_joint_deg(uint8_t enc_id) {
     if (enc_id >= 3) return 0.0f;
     int32_t count = encoder_states[enc_id].count * encoder_states[enc_id].direction;
     return ((float)count / (float)ENCODER_CPR_X4) * 360.0f;
 }
 
 static int32_t get_z_index_count(uint8_t enc_id) {
     if (enc_id >= 3) return 0;
     return encoder_states[enc_id].z_count;
 }
 
 static void reset_encoder_count(uint8_t enc_id) {
     if (enc_id >= 3) return;
     encoder_states[enc_id].count = 0;
 }
 
 static bool check_z_index_detected(uint8_t enc_id) {
     if (enc_id >= 3) return false;
     bool detected = g_z_index_detected[enc_id];
     g_z_index_detected[enc_id] = false;
     return detected;
 }
 
 // ============================================================================
 // UART
 // ============================================================================
 static QueueHandle_t uart_queue;
 
 static void uart_init(void) {
     uart_config_t uart_config = {
         .baud_rate = 115200,
         .data_bits = UART_DATA_8_BITS,
         .parity = UART_PARITY_DISABLE,
         .stop_bits = UART_STOP_BITS_1,
         .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
     };
     ESP_ERROR_CHECK(uart_param_config(UART_NUM_0, &uart_config));
     ESP_ERROR_CHECK(uart_set_pin(UART_NUM_0, 43, 44, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
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
 // CAN Send Functions
 // ============================================================================
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
 
static void send_input_torque(uint8_t node_id, float torque_nm) {
    uint8_t data[4];
    write_float_le(data, torque_nm);
    send_can_message(make_can_id(node_id, ODRIVE_CMD_SET_INPUT_TORQUE), data, 4);
}

// Set_Vel_Gains (cmd 0x01B): float vel_gain + float vel_integrator_gain (8 bytes).
// vel_gain = Nm / (rev/s), vel_integrator_gain = Nm / rev.
// Cao để motor mạnh thắng gravity load, thấp để motor yếu — tune tùy motor.
static void send_vel_gains(uint8_t node_id, float vel_gain, float vel_integrator_gain) {
    uint8_t data[8];
    write_float_le(&data[0], vel_gain);
    write_float_le(&data[4], vel_integrator_gain);
    send_can_message(make_can_id(node_id, ODRIVE_CMD_SET_VEL_GAINS), data, 8);
}
 
static void send_input_position(uint8_t node_id, float pos_rev, float vel_ff, float torque_ff) {
    uint8_t data[8];
    write_float_le(&data[0], pos_rev);
    int16_t vel_i = (int16_t)(vel_ff * 1000.0f);
    int16_t tau_i = (int16_t)(torque_ff * 1000.0f);
    memcpy(&data[4], &vel_i, 2);
    memcpy(&data[6], &tau_i, 2);
    send_can_message(make_can_id(node_id, ODRIVE_CMD_SET_INPUT_POS), data, 8);
}

// Set_Input_Vel (cmd 0x0D): float rev/s + float torque_ff (8 byte total).
// Dùng cho homing seek — motor chạy đúng tốc độ không bị drift tích lũy như position.
static void send_input_velocity(uint8_t node_id, float vel_rev_s, float torque_ff) {
    uint8_t data[8];
    write_float_le(&data[0], vel_rev_s);
    write_float_le(&data[4], torque_ff);
    send_can_message(make_can_id(node_id, ODRIVE_CMD_SET_INPUT_VEL), data, 8);
}
 
 static void send_axis_state(uint8_t node_id, uint32_t state) {
     uint8_t data[4];
     write_i32_le(data, (int32_t)state);
     send_can_message(make_can_id(node_id, ODRIVE_CMD_SET_AXIS_REQUESTED_STATE), data, 4);
 }
 
static void send_controller_mode(uint8_t node_id, int32_t control_mode, int32_t input_mode) {
    uint8_t data[8];
    write_i32_le(&data[0], control_mode);
    write_i32_le(&data[4], input_mode);
    send_can_message(make_can_id(node_id, ODRIVE_CMD_SET_CONTROLLER_MODES), data, 8);
}

// Set_Property (cmd 0x05) với endpoint float32: ghi 1 float vào controller config.
// Dùng cho: vel_limit, torque_ramp_rate, vel_ramp_rate, current_lim, etc.
// Endpoint IDs từ ODriveCAN.h (constants 0x00F, 0x015, ...).
static void send_controller_property_float(uint8_t node_id, uint16_t prop_endpoint, float value) {
    uint8_t data[8] = {0};
    // Property endpoint format: bits[15:5] = endpoint_id, bits[4:0] = type (1 = float32)
    uint16_t ep = (uint16_t)((prop_endpoint << 5) | 0x01);
    data[0] = (uint8_t)(ep & 0xFF);
    data[1] = (uint8_t)((ep >> 8) & 0xFF);
    write_float_le(&data[2], value);  // 4 bytes payload float32
    send_can_message(make_can_id(node_id, ODRIVE_CMD_SET_PROPERTY), data, 8);
}
 
 static void send_clear_errors(uint8_t node_id) {
     send_can_message(make_can_id(node_id, ODRIVE_CMD_CLEAR_ERRORS), NULL, 0);
 }
 
 static void send_get_encoder_estimates(uint8_t node_id) {
     send_can_message(make_can_id(node_id, ODRIVE_CMD_GET_ENCODER_ESTIMATES), NULL, 0);
 }
 
 static void send_torque_ramp_rate(uint8_t node_id, float rate) {
     uint8_t data[8] = {0};
     uint16_t endpoint = (uint16_t)((ODRIVE_PROP_TORQUE_RAMP_RATE << 5) | 0x01);
     data[0] = (uint8_t)(endpoint & 0xFF);
     data[1] = (uint8_t)((endpoint >> 8) & 0xFF);
     write_float_le(&data[2], rate);
     send_can_message(make_can_id(node_id, ODRIVE_CMD_SET_PROPERTY), data, 8);
 }
 
 // Set controller config property via CAN Set_Property (cmd 0x05)
 // endpoint_id is the 16-bit endpoint as defined by ODrive CAN protocol
 static void set_controller_property_u32(uint8_t node_id, uint16_t endpoint_id, uint32_t value) {
     uint8_t data[8] = {0};
     data[0] = (uint8_t)(endpoint_id & 0xFF);
     data[1] = (uint8_t)((endpoint_id >> 8) & 0xFF);
     data[2] = (uint8_t)(value & 0xFF);
     data[3] = (uint8_t)((value >> 8) & 0xFF);
     data[4] = (uint8_t)((value >> 16) & 0xFF);
     data[5] = (uint8_t)((value >> 24) & 0xFF);
     send_can_message(make_can_id(node_id, ODRIVE_CMD_SET_PROPERTY), data, 8);
 }
 
 static void set_controller_property_float(uint8_t node_id, uint16_t endpoint_id, float value) {
     uint8_t data[8] = {0};
     data[0] = (uint8_t)(endpoint_id & 0xFF);
     data[1] = (uint8_t)((endpoint_id >> 8) & 0xFF);
     write_float_le(&data[2], value);
     send_can_message(make_can_id(node_id, ODRIVE_CMD_SET_PROPERTY), data, 8);
 }
 

 static void auto_apply_torque_mode(void) {
     uart_write_str("INFO: Auto-applying torque mode (control_mode=1, input_mode=6, ramp=3 Nm/s)\n");
     for (int i = 0; i < 3; i++) {
         send_clear_errors(i);
         send_controller_mode(i, CONTROL_MODE_TORQUE_CONTROL, INPUT_MODE_TORQUE_RAMP);
         // Set torque_ramp_rate via property 0x015
         uint8_t data[8] = {0};
         uint16_t endpoint = (uint16_t)((ODRIVE_PROP_TORQUE_RAMP_RATE << 5) | 0x01);
        data[0] = (uint8_t)(endpoint & 0xFF);
        data[1] = (uint8_t)((endpoint >> 8) & 0xFF);
        write_float_le(&data[2], 3.0f);
        send_can_message(make_can_id(i, ODRIVE_CMD_SET_PROPERTY), data, 8);
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
 
 static void process_command(const char* cmd) {
     int node_id;
     float f_val, pos, vel, tau;
     int i_val, i_val2;
 
     if (strcmp(cmd, "PING") == 0) {
         uart_write_str("PONG\n");
     }
     else if (strcmp(cmd, "HELP") == 0) {
         uart_write_str("Commands (bridge mode - PC tính CTC):\n");
         uart_write_str("  Legacy: TORQUE/POS/STATE/MODE/RAMP/CLEAR/CLOSE/IDLE/HOME/STATUS\n");
         uart_write_str("  Short:  T{id}:Nm   P{id}:rev,rev_s   GOTO q0 q1 q2 (deg)\n");
         uart_write_str("          HOLD   GAIN kp0 kp1 kp2 kd0 kd1 kd2   VMAX vmax\n");
         uart_write_str("          PRISM hip_mm knee_mm   CYL joint dir   LOCK l0 l1 l2\n");
         uart_write_str("          DX:dx1,dx2   LP fc_vel [fc_set]\n");
         uart_write_str("          HOME [<q_max_deg>]   HOMECANCEL   ENC   PING\n");
         uart_write_str("          HOME knee: seek EXT(GPIO41) → KOK,<q_max_deg>  (count reset to 0)\n");
     }
     else if (strcmp(cmd, "STATUS") == 0) {
         uart_write_fmt("STATUS: TX=%lu fail=%lu RX=%lu HB=[%d,%d,%d] CL=%d motion=%d\n",
             can_tx_count, can_tx_fail, can_rx_count,
             hb_received[0], hb_received[1], hb_received[2],
             closed_loop_state ? 1 : 0,
             motion_active ? 1 : 0);
     }
     else if (strcmp(cmd, "ENC") == 0) {
         uart_write_fmt("ENC: %d,%d,%d | %.2f,%.2f,%.2f deg | Z:%d,%d,%d\n",
             encoder_states[0].count, encoder_states[1].count, encoder_states[2].count,
             get_joint_deg(0), get_joint_deg(1), get_joint_deg(2),
             encoder_states[0].z_count, encoder_states[1].z_count, encoder_states[2].z_count);
     }
     else if (strcmp(cmd, "CLOSE") == 0) {
         for (int i = 0; i < 3; i++) {
             send_clear_errors(i);
             send_controller_mode(i, CONTROL_MODE_TORQUE_CONTROL, INPUT_MODE_TORQUE_RAMP);
             // Set torque_ramp_rate (endpoint 0x015) via Set_Property
             uint8_t data[8] = {0};
             uint16_t endpoint = (uint16_t)((ODRIVE_PROP_TORQUE_RAMP_RATE << 5) | 0x01);  // type=1 float
             data[0] = (uint8_t)(endpoint & 0xFF);
             data[1] = (uint8_t)((endpoint >> 8) & 0xFF);
             write_float_le(&data[2], 3.0f);  // 3 Nm/s ramp (giam tu 10 de tranh SPINOUT o Hall encoder)
             send_can_message(make_can_id(i, ODRIVE_CMD_SET_PROPERTY), data, 8);
             vTaskDelay(pdMS_TO_TICKS(5));
         }
         vTaskDelay(pdMS_TO_TICKS(50));
         for (int i = 0; i < 3; i++) {
             send_axis_state(i, AXIS_STATE_CLOSED_LOOP_CONTROL);
             vTaskDelay(pdMS_TO_TICKS(5));
         }
         closed_loop_state = true;
         uart_write_str("INFO: CLOSE (torque_ramp=10 Nm/s via property 0x015)\n");
     }
    else if (strcmp(cmd, "IDLE") == 0) {
        for (int i = 0; i < 3; i++) {
            send_axis_state(i, AXIS_STATE_IDLE);
            vTaskDelay(pdMS_TO_TICKS(5));
        }
        closed_loop_state = false;
        motion_active = false;
        for (int i = 0; i < 3; i++) last_torque_nm[i] = 0.0f;
        // Reset LP vel state để tránh giá trị cũ khi khởi động motion mới.
        for (int i = 0; i < 3; i++) vel_lp_prev[i] = 0.0f;
        uart_write_str("INFO: IDLE\n");
    }
    else if (strcmp(cmd, "HOME") == 0 || strcmp(cmd, "HOME 1") == 0) {
        // Home khớp knee (axis 1) — dùng q_max mặc định 90°.
        home_q_max_deg = 90.0f;
        home_pending = true;
        motion_active = false;
        vel_lp_prev[HOME_KNEE_AXIS] = 0.0f;
        uart_write_fmt("INFO: HOME knee (q_max=%.1f°)\n", home_q_max_deg);
    }
    else if (sscanf(cmd, "HOME %f", &f_val) == 1) {
        // HOME <q_max_deg>: đặt ROM khớp knee rồi start homing.
        // Dùng khi anh muốn khai báo ROM khác 90° (vd 120° cho knee co sâu hơn).
        home_q_max_deg = f_val;
        home_pending = true;
        motion_active = false;
        vel_lp_prev[HOME_KNEE_AXIS] = 0.0f;
        uart_write_fmt("INFO: HOME knee (q_max=%.1f°) start\n", home_q_max_deg);
    }
    else if (sscanf(cmd, "HOME %d", &node_id) == 1 && node_id >= 0 && node_id < 3) {
        if (node_id != HOME_KNEE_AXIS) {
            uart_write_fmt("WARN: HOME axis=%d not supported (only knee=1)\n", node_id);
        } else {
            home_pending = true;
            motion_active = false;
            vel_lp_prev[node_id] = 0.0f;
            uart_write_fmt("INFO: HOME axis=%d\n", node_id);
        }
    }
    else if (strcmp(cmd, "HOMECANCEL") == 0) {
        // Hủy homing đang chạy. Brake bằng vel=0 (mode hiện tại là velocity VEL_RAMP).
        home_pending = false;
        home_state = HOME_IDLE;
        send_input_velocity(HOME_KNEE_AXIS, 0.0f, 0.0f);
        uart_write_str("INFO: HOMECANCEL\n");
    }
    // ── MODE <node> <ctrl> <input>: switch ODrive controller/input mode ───────
    //   control_mode: 0=voltage, 1=torque, 2=velocity, 3=position
    //   input_mode:   0=inactive, 1=passthrough, 2=vel_ramp, 3=pos_filter,
    //                 4=mix_ch, 5=trap_traj, 6=torque_ramp, 7=mirror
    // Dùng để chuyển knee từ torque (lúc HOME seek) sang position (giữ EXT sau HOME).
    else if (sscanf(cmd, "MODE %d %d %d", &node_id, &i_val, &i_val2) == 3) {
        if (node_id >= 0 && node_id < 3) {
            send_controller_mode((uint8_t)node_id, i_val, i_val2);
            uart_write_fmt("[ESP32] MODE node=%d ctrl=%d input=%d\n",
                           node_id, i_val, i_val2);
        }
    }
    else if (sscanf(cmd, "TORQUE %d %f", &node_id, &f_val) == 2) {
         if (node_id >= 0 && node_id < 3) {
             send_input_torque(node_id, f_val);
             // DEBUG: confirm CAN tx reached ODrive
             uart_write_fmt("[ESP32] TORQUE sent -> CAN node=%d cmd=0x%02X torque=%.6f Nm\n",
                            node_id, ODRIVE_CMD_SET_INPUT_TORQUE, f_val);
         }
     }
     else if (sscanf(cmd, "POS %d %f %f %f", &node_id, &pos, &vel, &tau) == 4) {
         if (node_id >= 0 && node_id < 3) {
             send_input_position(node_id, pos, vel, tau);
         }
     }
     else if (sscanf(cmd, "STATE %d %d", &node_id, &i_val) == 2) {
         if (node_id >= 0 && node_id < 3) {
             send_axis_state(node_id, (uint32_t)i_val);
         }
     }
     else if (sscanf(cmd, "MODE %d %d %d", &node_id, &i_val, &i_val2) == 3) {
         if (node_id >= 0 && node_id < 3) {
             send_controller_mode(node_id, i_val, i_val2);
         }
     }
     else if (sscanf(cmd, "RAMP %d %f", &node_id, &f_val) == 2) {
         if (node_id >= 0 && node_id < 3) {
             send_torque_ramp_rate(node_id, f_val);
             uart_write_fmt("TX: RAMP %d %.1f\n", node_id, f_val);
         }
     }
     else if (sscanf(cmd, "CLEAR %d", &node_id) == 1) {
         if (node_id >= 0 && node_id < 3) {
             send_clear_errors(node_id);
         }
     }
    // ── Short-form torque: T{id}:Nm ──────────────────────────────────────
    else if (sscanf(cmd, "T%d:%f", &node_id, &f_val) == 2) {
        if (node_id >= 0 && node_id < 3) {
            if (locked_axes[node_id]) {
                f_val = 0.0f;
            }
            // Knee (axis 1) có thể đang ở position mode (sau HOME) → torque cmd bị ignore.
            // Auto-switch về torque mode để khớp với PC đang bridge mode CTC.
            if (node_id == HOME_KNEE_AXIS) {
                send_controller_mode((uint8_t)node_id,
                                     CONTROL_MODE_TORQUE_CONTROL, INPUT_MODE_TORQUE_RAMP);
                vTaskDelay(pdMS_TO_TICKS(10));
            }
            // Mirror PC torque để trả về feedback (dùng cho logging/GUI)
            last_torque_nm[node_id] = f_val;
            last_pc_cmd_us = esp_timer_get_time();
            // Forward qua CAN - ODrive sẽ ramp tới giá trị này
            send_input_torque((uint8_t)node_id, f_val);
        }
    }
    // ── Short-form position: P{id}:pos,vel ───────────────────────────────
    else if (sscanf(cmd, "P%d:%f,%f", &node_id, &pos, &vel) == 3) {
        if (node_id >= 0 && node_id < 3) {
            if (locked_axes[node_id]) {
                pos = motor_pos[node_id];  // giữ nguyên
                vel = 0.0f;
            }
            // Knee (axis 1): đảm bảo position mode (passthrough) để pos cmd có hiệu lực.
            if (node_id == HOME_KNEE_AXIS) {
                send_controller_mode((uint8_t)node_id,
                                     CONTROL_MODE_POSITION_CONTROL, INPUT_MODE_PASSTHROUGH);
                vTaskDelay(pdMS_TO_TICKS(10));
            }
            last_position_rev[node_id] = pos;
            last_pc_cmd_us = esp_timer_get_time();
            // Forward: pos_rev, vel_ff=0, torque_ff=0 (PC đã tính CTC ở đâu đó)
            send_input_position((uint8_t)node_id, pos, vel, 0.0f);
        }
    }
     // ── GOTO target (deg) - đánh dấu motion_active để PC biết ──────────
     else if (sscanf(cmd, "GOTO %f %f %f", &pos, &vel, &tau) == 3) {
         // PC đã tính trajectory, đây là signal để ESP32 bật motion_active flag
         motion_active = true;
         last_pc_cmd_us = esp_timer_get_time();
         uart_write_str("INFO: GOTO received\n");
     }
     // ── HOLD: dừng motion ───────────────────────────────────────────────
     else if (strcmp(cmd, "HOLD") == 0) {
         motion_active = false;
         for (int i = 0; i < 3; i++) {
             last_torque_nm[i] = 0.0f;
             if (closed_loop_state) {
                 send_input_torque(i, 0.0f);
             }
         }
         uart_write_str("INFO: HOLD\n");
     }
     // ── GAIN: lưu Kp/Kd để debug (bridge mode không dùng trực tiếp) ────
     else if (sscanf(cmd, "GAIN %f %f %f %f %f %f",
                     &f_val, &pos, &vel, &tau, &f_val, &pos) >= 3) {
         // Chỉ echo lại để PC xác nhận; PC tự tính CTC
         uart_write_fmt("INFO: GAIN echo kp=(%.2f,%.2f,%.2f) kd=(%.2f,%.2f,%.2f)\n",
                        f_val, pos, vel, tau, f_val, pos);
     }
     // ── VMAX: lưu max velocity (echo only) ──────────────────────────────
     else if (sscanf(cmd, "VMAX %f", &f_val) == 1) {
         uart_write_fmt("INFO: VMAX=%.2f\n", f_val);
     }
     // ── PRISM: cập nhật chiều dài prismatic (mm) - echo only ────────────
     else if (sscanf(cmd, "PRISM %f %f", &f_val, &pos) == 2) {
         // Bridge không tự tính - chỉ echo để PC xác nhận
         uart_write_fmt("PRISM %.1f %.1f\n", f_val, pos);
     }
     // ── CYL: điều khiển cylinder (chưa có driver thực) ──────────────────
     else if (sscanf(cmd, "CYL %d %d", &node_id, &i_val) == 2) {
         uart_write_fmt("WARN: CYL not implemented (joint=%d dir=%d)\n", node_id, i_val);
     }
     // ── LOCK: khóa axes không nhận torque ───────────────────────────────
     else if (sscanf(cmd, "LOCK %d %d %d", &i_val, &i_val2, &node_id) == 3) {
         for (int i = 0; i < 3; i++) {
             // i_val -> axis0, i_val2 -> axis1, node_id -> axis2
             int v = (i == 0) ? i_val : (i == 1) ? i_val2 : node_id;
             locked_axes[i] = (v != 0);
         }
         uart_write_fmt("INFO: LOCK [%d,%d,%d]\n", locked_axes[0], locked_axes[1], locked_axes[2]);
     }
     // ── DX: cập nhật COM translation (echo only - CTC chạy trên PC) ────
     else if (sscanf(cmd, "DX:%f,%f", &f_val, &pos) == 2) {
         uart_write_fmt("INFO: DX dx1=%.4f dx2=%.4f\n", f_val, pos);
     }
     // ── LP: cập nhật LP filter cutoff (echo only) ──────────────────────
     else if (sscanf(cmd, "LP %f", &f_val) == 1) {
         uart_write_fmt("INFO: LP fc=%.1f\n", f_val);
     }
     // ── S,motion: PC set motion_active trực tiếp (fallback) ────────────
     else if (sscanf(cmd, "S,%d", &i_val) == 1) {
         motion_active = (i_val != 0);
         uart_write_fmt("INFO: S motion=%d\n", motion_active ? 1 : 0);
     }
     else if (strlen(cmd) > 0 && cmd[0] != '\r' && cmd[0] != '\n') {
         uart_write_fmt("WARN: unknown cmd: %s\n", cmd);
     }
 }
 
 static void uart_task(void* arg) {
     char cmd_buf[256];
     int buf_idx = 0;
     
     while (1) {
         uint8_t ch;
         if (uart_read_bytes(UART_NUM_0, &ch, 1, portMAX_DELAY) == 1) {
             if (ch == '\n' || ch == '\r') {
                 cmd_buf[buf_idx] = '\0';
                 if (buf_idx > 0) {
                     process_command(cmd_buf);
                     buf_idx = 0;
                 }
             } else if (buf_idx < (int)(sizeof(cmd_buf) - 1)) {
                 cmd_buf[buf_idx++] = ch;
             }
         }
     }
 }
 
 // ============================================================================
 // CAN RX Processing
 // ============================================================================
 static void process_can_rx(void) {
     twai_message_t rx_msg;
     
     while (twai_receive(&rx_msg, pdMS_TO_TICKS(0)) == ESP_OK) {
         can_rx_count++;
         uint8_t node_id = get_node_id(rx_msg.identifier);
         uint8_t cmd = get_cmd_id(rx_msg.identifier);
         
         if (node_id < 3) {
             if (cmd == ODRIVE_CMD_HEARTBEAT && rx_msg.data_length_code >= 7) {
                 uint16_t state = rx_msg.data[4] | (rx_msg.data[5] << 8);
                 uint8_t req_state = rx_msg.data[6];
                 axis_state[node_id] = state;
                 hb_received[node_id] = true;
                 // Heartbeat is sampled by bridge_task via axis_state[]. Do NOT
                 // spam "HB" lines here -- they collide with the CSV feedback
                 // frame and confuse the Python parser.
             }
             else if (cmd == ODRIVE_CMD_GET_ENCODER_ESTIMATES && rx_msg.data_length_code >= 8) {
                 float pos_est = read_float_le(rx_msg.data);
                 float vel_est = read_float_le(rx_msg.data + 4);
                 motor_pos[node_id] = pos_est;
                 motor_vel[node_id] = vel_est;
                 // Encoder estimates are streamed by bridge_task in CSV form
                 // (FB,mot0,mot1,mot2,j0,j1,j2,tau0,tau1,tau2,motion). Do NOT
                 // emit a per-frame "FB %d ..." line here.
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
             uart_write_str("WARN: CAN bus off, recovering...\n");
             twai_initiate_recovery();
             vTaskDelay(pdMS_TO_TICKS(250));
         }
     }
     if (alerts & TWAI_ALERT_BUS_RECOVERED) {
         twai_start();
         bus_off_recovering = false;
         uart_write_str("INFO: CAN recovered\n");
     }
     if (alerts & TWAI_ALERT_TX_FAILED) {
         can_tx_fail++;
     }
 }
 
 // ============================================================================
 // CAN Init
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
 
// ============================================================================
// Bridge Task
// ============================================================================
static void bridge_task(void* arg) {
    int64_t last_enc_req_us = 0;
    int64_t last_fb_us = 0;
    int64_t last_status_us = 0;

    float last_joint_deg[3] = {0.0f, 0.0f, 0.0f};
    // Vél đã qua LP-filter (deg/s) — gửi lên PC thay vì vel thô để tránh
    // spike khi motor đứng yên + 1 LSB drift.
    float vel_deg_s[3] = {0.0f, 0.0f, 0.0f};
    int64_t last_fb_dt_us = 10000;   // dt cho lần đầu (10ms nominal)

    while (1) {
        int64_t now_us = esp_timer_get_time();

        process_can_rx();
        check_can_health();

        // Request encoder estimates at 50Hz
        if (now_us - last_enc_req_us >= 20000) {
            last_enc_req_us = now_us;
            send_get_encoder_estimates(ODRV0_NODE_ID);
            send_get_encoder_estimates(ODRV1_NODE_ID);
            send_get_encoder_estimates(ODRV2_NODE_ID);
        }

        // Send CSV feedback at 100Hz:
        //   FB,mot0_rev,mot1_rev,mot2_rev,j0_deg,j1_deg,j2_deg,tau0_Nm,tau1_Nm,tau2_Nm,motion,vel0,vel1,vel2
        // vel0..2 đã LP-filter (1-pole IIR, fc=50Hz) để loại spike khi encoder
        // drift 1 LSB ở tốc độ thấp. PC vẫn giữ LP riêng (fc=80Hz) làm tầng mượt.
        if (now_us - last_fb_us >= 10000) {
            last_fb_dt_us = (last_fb_us == 0) ? 10000 : (now_us - last_fb_us);
            last_fb_us = now_us;

            float joint_deg[3];
            for (int i = 0; i < 3; i++) {
                joint_deg[i] = get_joint_deg(i);
                // dt thực tế (giây) thay vì cứng 0.01 để tránh bias khi loop bị trễ.
                float dt_s = (float)last_fb_dt_us / 1e6f;
                if (dt_s < 1e-4f) dt_s = 1e-4f;   // guard chia 0
                float vel_inst = (joint_deg[i] - last_joint_deg[i]) / dt_s;
                // 1-pole LP: y[n] = α·x[n] + (1-α)·y[n-1]
                vel_deg_s[i] = vel_lp_alpha * vel_inst + (1.0f - vel_lp_alpha) * vel_lp_prev[i];
                vel_lp_prev[i] = vel_deg_s[i];
                last_joint_deg[i] = joint_deg[i];
            }

           uart_write_fmt("FB,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%d,%.3f,%.3f,%.3f\n",
               motor_pos[0], motor_pos[1], motor_pos[2],
               joint_deg[0], joint_deg[1], joint_deg[2],
               last_torque_nm[0], last_torque_nm[1], last_torque_nm[2],
               motion_active ? 1 : 0,
               vel_deg_s[0], vel_deg_s[1], vel_deg_s[2]);
        }
 
// Watchdog: nếu PC im lặng > 500ms trong closed_loop, gửi torque=0 để brake
        // SKIP nếu đang homing (home_task đang điều khiển motor trực tiếp).
        if (closed_loop_state &&
            last_pc_cmd_us > 0 &&
            (now_us - last_pc_cmd_us) > 500000 &&
            home_state == HOME_IDLE) {
             for (int i = 0; i < 3; i++) {
                 if (last_torque_nm[i] != 0.0f) {
                     last_torque_nm[i] = 0.0f;
                     send_input_torque(i, 0.0f);
                 }
             }
             motion_active = false;
         }
         
         // Status every 5s
         if (now_us - last_status_us >= 5000000) {
             last_status_us = now_us;
             uart_write_fmt("INFO: CAN TX=%lu RX=%lu\n", can_tx_count, can_rx_count);
         }
         
         vTaskDelay(pdMS_TO_TICKS(1));
     }
 }
 
// ============================================================================
// Homing task — Knee joint (axis 1) bằng 1 limit switch (EXT = duỗi)
// ----------------------------------------------------------------------------
// Flow:
//   IDLE → (pending) → SEEK_EXT (quay dir=+1 cùng chiều moment dương, chờ GPIO 41 pressed)
//       → DONE: brake, reset count=0 tại EXT, gửi "KOK,<q_max>"
//
// q_max_deg được PC cung cấp qua command "HOME <q_max_deg>" (mặc định 90°).
//
// Sau calib (PC side):
//   count=0 tại EXT (duỗi thẳng) → q_feedback = 0° khi duỗi
//   Gập về 1 phía → q_feedback = -q_max (negative về phía flex).
//                      = -q_max                  (khi encoder_raw = phi_ext + q_max)
//
// Chạy ở 100Hz trên core 0, priority 8 (thấp hơn bridge_task 10).
// ============================================================================
static void home_task(void* arg) {
    const TickType_t period = pdMS_TO_TICKS(10);   // 100 Hz
    TickType_t last = xTaskGetTickCount();
    const int axis = HOME_KNEE_AXIS;  // = 1

    while (1) {
        int64_t now_us = esp_timer_get_time();

        // ── Request mới? ──
        if (home_pending) {
            home_pending = false;
            // Safety: nếu ODrive chưa ở closed-loop, brake và bail.
            if (!closed_loop_state) {
                uart_write_str("ERR: HOME rejected — ODrive chưa ở CLOSED_LOOP (gửi CLOSE trước)\n");
                home_state = HOME_IDLE;
                break;
            }
            // Strategy: VELOCITY mode + vel_gain cao + INPUT_MODE_VEL_RAMP.
            //   - vel_gain=1.0, vel_int_gain=4.0 (default 0.166 → quá yếu, stall dưới gravity).
            //   - VEL_RAMP: ODrive tự ramp vel 0→target mượt theo vel_ramp_rate = 2 rev/s/s.
            //     0 → 0.1 rev/s mất 50ms → mượt, không giật.
            //   - BỎ torque mode (anh báo "giật rất mạnh" khi torque > gravity moment).
            //            //            send_vel_gains(axis, HOME_SEEK_VEL_GAIN, HOME_SEEK_VEL_INT_GAIN);
            vTaskDelay(pdMS_TO_TICKS(20));

            // vel_limit, vel_ramp_rate đã config cứng trong ODrive_config.py:
            //   vel_limit = 5.0 rad/s (an toàn, tránh overshoot)
            //   vel_ramp_rate = 1.0 (mượt)
            //   max_regen_current = 10A (chịu brake đột ngột)
            // KHÔNG ghi đè ở đây — endpoint IDs thay đổi theo firmware version.

            send_controller_mode(axis, CONTROL_MODE_VELOCITY_CONTROL, INPUT_MODE_VEL_RAMP);
            vTaskDelay(pdMS_TO_TICKS(20));  // cho ODrive áp dụng mode
            home_state = HOME_KNEE_SEEK_EXT;
            home_start_us = now_us;
            home_motor_dir = +1;  // toward EXT (duỗi) — cùng chiều vel dương
            motion_active = true;
            last_pc_cmd_us = now_us;  // reset watchdog
            uart_write_fmt("INFO: HOME knee start (count=%ld) → seek EXT vel=+%.2f rev/s (vg=%.1f, vig=%.1f)\n",
                           encoder_states[axis].count, HOME_SEEK_VEL_REV_S,
                           HOME_SEEK_VEL_GAIN, HOME_SEEK_VEL_INT_GAIN);
        }

        // ── State machine ──
        switch (home_state) {
        case HOME_IDLE:
            break;

        case HOME_KNEE_SEEK_EXT: {
            // Timeout check
            int64_t elapsed_ms = (now_us - home_start_us) / 1000;
            if (elapsed_ms > HOME_TIMEOUT_MS) {
                uart_write_fmt("ERR: HOME knee timeout after %lldms (chưa chạm GPIO41)\n",
                               elapsed_ms);
                send_input_velocity(axis, 0.0f, 0.0f);
                home_state = HOME_IDLE;
                motion_active = false;
                break;
            }

            // Check switch EXT
            if (limit_ext_is_pressed()) {
                home_phi_extended = encoder_states[axis].count;
                uart_write_fmt("INFO: HIT EXT (GPIO41) count=%ld (q_max=%.1f°)\n",
                               home_phi_extended, home_q_max_deg);
                send_input_velocity(axis, 0.0f, 0.0f);  // brake (ramp về 0 mượt)
                home_state = HOME_KNEE_DONE;
                motion_active = false;
                break;
            }

            // VEL_RAMP mode: ODrive tự ramp vel 0→target mượt theo vel_ramp_rate.
            // Chỉ cần gửi target 1 lần, ODrive giữ và tự ramp.
            // vel_gain cao → motor mạnh thắng gravity (không stall).
            send_input_velocity(axis, HOME_SEEK_VEL_REV_S * home_motor_dir, 0.0f);
            motion_active = true;
            last_pc_cmd_us = now_us;  // reset watchdog (home_task đang drive motor)
            break;
        }

        case HOME_KNEE_DONE: {
            // Reset encoder count về 0 tại vị trí EXT (home) TRƯỚC, để count=0 = q_feedback=0.
            encoder_states[axis].count = 0;
            vel_lp_prev[axis] = 0.0f;

            // Gửi KOK cho PC: chỉ cần q_max_deg (vì PC biết EXT = 0° theo hệ q_feedback).
            // PC sẽ set isOffset=True, q_max_deg=q_max, khi duỗi q_feedback=0°.
            uart_write_fmt("KOK,%.3f\n", home_q_max_deg);
            uart_write_fmt("INFO: HOME knee done → q_max=%.1f°, count reset to 0\n",
                           home_q_max_deg);

            // Brake motor tại EXT bằng vel=0 trước khi switch mode.
            send_input_velocity(axis, 0.0f, 0.0f);
            vTaskDelay(pdMS_TO_TICKS(50));  // đợi vel ramp về 0

            // CHUYỂN KNEE sang position mode (passthrough) để HOLD chắc tại EXT.
            // Sau khi home, motor phải đứng yên chống trôi do trọng lực/load.
            send_controller_mode(axis, CONTROL_MODE_POSITION_CONTROL, INPUT_MODE_PASSTHROUGH);
            vTaskDelay(pdMS_TO_TICKS(20));  // cho ODrive áp dụng mode

            // Hold vị trí EXT (0 rev, vel=0) bằng position mode.
            // ODrive sẽ tự P-gain giữ encoder tại count=0 → motor không drift.
            send_input_position(axis, 0.0f, 0.0f, 0.0f);
            last_position_rev[axis] = 0.0f;
            last_pc_cmd_us = esp_timer_get_time();  // reset watchdog

            home_state = HOME_IDLE;
            motion_active = false;
            break;
        }
        }

        vTaskDelayUntil(&last, period);
    }
}

// ============================================================================
// Main
// ============================================================================
void app_main(void) {
    uart_init();
    encoder_init();
    
    if (!twai_init()) {
        uart_write_str("ERROR: CAN init failed\n");
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    twai_initialized = true;
    
    uart_write_str("\n===========================================\n");
    uart_write_str("ESP32-S3 ODrive CAN Bridge\n");
    uart_write_str("Bridge mode - PC tính CTC, ESP32 forward torque/pos\n");
    uart_write_str("===========================================\n");
    uart_write_str("CAN: 250Kbit/s\n");
    uart_write_fmt("ENC: %d CPR (x4), A/B/Z\n", ENCODER_CPR_X4);
    uart_write_str("Homing: knee EXT switch (GPIO41, active LOW) — HOME [q_max_deg] / HOMECANCEL\n");
    uart_write_str("READY\n\n");

    // Auto-apply torque mode on boot so that torque commands work without
    // needing a separate "CLOSE" command. This overwrites ODrive's default
    // control_mode=3 (position) which makes Set_Input_Torque a no-op.
    auto_apply_torque_mode();
    
    xTaskCreatePinnedToCore(uart_task, "uart", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(bridge_task, "bridge", 4096, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(home_task, "home", 4096, NULL, 8, NULL, 0);
    
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}