#include "rs05_motor_driver.hpp"
#include "timer.hpp"
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>

RS05MotorDriver::RS05MotorDriver(uint16_t motor_id, const std::string& can_interface, 
                                 uint16_t master_id)
    : MotorDriver(), can_interface_(can_interface), master_id_(master_id) {
    motor_id_ = motor_id;
    can_ = SocketCAN::get(can_interface);
    
    // 注册回调（私有协议使用扩展帧）
    auto callback = std::bind(&RS05MotorDriver::canRxCallback, this, 
                             std::placeholders::_1);
    can_->add_can_callback(callback, motor_id_, true);  // true = 扩展帧
}

RS05MotorDriver::~RS05MotorDriver() {
    can_->remove_can_callback(motor_id_);
}

uint32_t RS05MotorDriver::buildCanID(uint8_t comm_type, uint16_t data_field) const {
    uint32_t comm_type_bits = (comm_type & 0x1F) << 24;
    uint32_t data_field_bits = (data_field & 0xFFFF) << 8;
    uint32_t id_bits = motor_id_ & 0xFF;
    return comm_type_bits | data_field_bits | id_bits;
}

uint16_t RS05MotorDriver::floatToUint(float x, float x_min, float x_max, int bits) const {
    float span = x_max - x_min;
    float offset = x_min;
    x = std::max(std::min(x, x_max), x_min);
    return static_cast<uint16_t>((x - offset) * ((1 << bits) - 1) / span);
}

float RS05MotorDriver::uintToFloat(uint16_t x_int, float x_min, float x_max, int bits) const {
    float span = x_max - x_min;
    float offset = x_min;
    uint16_t max_val = (1 << bits) - 1;
    if (max_val == 0) return offset;
    return static_cast<float>(x_int * span / max_val + offset);
}

void RS05MotorDriver::sendCommand(uint8_t comm_type, uint16_t data_field, const uint8_t* payload) {
    uint32_t real_can_id = buildCanID(comm_type, data_field);
    
    can_frame frame;
    frame.can_id = real_can_id | CAN_EFF_FLAG;  // 扩展帧
    frame.can_dlc = 8;
    std::memcpy(frame.data, payload, 8);
    
    if (can_->transmit(frame)) {
        response_count_++;
    }
}

void RS05MotorDriver::canRxCallback(const can_frame& rx_frame) {
    // 静态计数器，用于控制打印频率
    static int rs05_print_count = 0;

    // 检查帧类型：扩展帧(29位) vs 标准帧(11位)
    bool is_extended = (rx_frame.can_id & CAN_EFF_FLAG);

    if (is_extended) {
        // === 通信类型2：电机反馈数据（扩展帧，29位ID） ===
        uint32_t can_id = rx_frame.can_id & CAN_EFF_MASK;
        uint8_t comm_type = (can_id >> 24) & 0x1F;  // 通信类型 (bit28~24)

        // 根据通信类型分别处理
        switch (comm_type) {
            case 0x02: {  // 通信类型2：电机反馈数据
                // 私有协议反馈帧：[Type][Status/Fault][MotorID][MasterID]
                // MotorID 位于 bit 8-15；不兼容低 8 位命令帧布局，避免误解析。
                uint8_t motor_id = (can_id >> 8) & 0xFF;
                if (motor_id != motor_id_) return;

                // 检查是否为版本号查询的返回帧
                if (version_query_pending_.load()) {
                    // 版本号查询返回帧：特殊处理
                    version_query_pending_.store(false);
                    // 版本号数据在Byte1-6，如0xC4, 0x56等
                    printf("[RS05 Motor %d] Version: %02X %02X %02X %02X %02X %02X\n",
                           motor_id_, rx_frame.data[1], rx_frame.data[2], rx_frame.data[3],
                           rx_frame.data[4], rx_frame.data[5], rx_frame.data[6]);
                    return;  // 不更新运动状态
                }

                // --- 解析数据 (通信类型2：16位运动数据格式) ---
                uint16_t pos_uint = (rx_frame.data[0] << 8) | rx_frame.data[1];      // 角度 [0~65535]
                uint16_t vel_uint = (rx_frame.data[2] << 8) | rx_frame.data[3];      // 速度 [0~65535]
                uint16_t torque_uint = (rx_frame.data[4] << 8) | rx_frame.data[5];   // 力矩 [0~65535]
                uint16_t temp_raw = (rx_frame.data[6] << 8) | rx_frame.data[7];      // 温度 Temp*10

                // 故障信息来自 CAN ID 的 data_field 高 6 位
                uint16_t data_field = (can_id >> 8) & 0xFFFF;
                uint8_t fault_info = (data_field >> 8) & 0x3F;

                // 解算物理量（16位格式）
                motor_pos_ = uintToFloat(pos_uint, P_MIN, P_MAX, 16);
                motor_spd_ = uintToFloat(vel_uint, V_MIN, V_MAX, 16);
                motor_current_ = uintToFloat(torque_uint, T_MIN, T_MAX, 16);
                motor_temperature_ = static_cast<float>(temp_raw) / 10.0f;

                error_id_ = fault_info;
                break;
            }

            case 0x00: {  // 通信类型0：设备ID反馈
                // 64位MCU唯一标识符
                uint64_t uid = 0;
                for (int i = 0; i < 8; i++) {
                    uid |= ((uint64_t)rx_frame.data[i] << (i * 8));
                }
                printf("[RS05 Motor %d] Device UID: 0x%016lX\n", motor_id_, uid);
                break;
            }

            case 0x11: {  // 通信类型17：单参数读取反馈
                uint16_t param_index = (rx_frame.data[0] << 8) | rx_frame.data[1];
                // 参数值在Byte4-7，根据参数类型可能是float或int
                uint32_t param_value = (rx_frame.data[4] << 24) | (rx_frame.data[5] << 16) |
                                     (rx_frame.data[6] << 8) | rx_frame.data[7];
                printf("[RS05 Motor %d] Parameter 0x%04X = 0x%08X\n", motor_id_, param_index, param_value);
                break;
            }

            default:
                // 其他通信类型暂时跳过
                break;
        }

    } else {
        // === 应答指令1：控制指令应答帧（标准帧，11位ID） ===
        uint32_t can_id = rx_frame.can_id & CAN_SFF_MASK;

        // 应答指令1使用主机ID (0xFD)
        if (can_id != 0xFD) return;

        // Byte0应为电机CAN ID
        uint8_t motor_id = rx_frame.data[0];
        if (motor_id != motor_id_) return;

        // --- 解析数据 (应答指令1：混合数据格式) ---
        uint16_t pos_uint = (rx_frame.data[1] << 8) | rx_frame.data[2];     // 角度 [0~65535] (16位)

        // 速度：12位数据分散存储
        // Byte3为高8位，Byte4[7-4]为低4位
        uint16_t vel_high = rx_frame.data[3];
        uint8_t vel_low_high = (rx_frame.data[4] >> 4) & 0x0F;  // Byte4高4位
        uint16_t vel_uint = (vel_high << 4) | vel_low_high;      // [0~4095] (12位)

        // 力矩：12位数据分散存储
        // Byte4[3-0]为高4位，Byte5为低8位
        uint8_t torque_high_low = rx_frame.data[4] & 0x0F;      // Byte4低4位
        uint16_t torque_low = rx_frame.data[5];
        uint16_t torque_uint = (torque_high_low << 8) | torque_low;  // [0~4095] (12位)

        uint16_t temp_raw = (rx_frame.data[6] << 8) | rx_frame.data[7];     // 温度 Temp*10 (16位)

        // 解算物理量（混合格式）
        motor_pos_ = uintToFloat(pos_uint, P_MIN, P_MAX, 16);
        motor_spd_ = uintToFloat(vel_uint, V_MIN, V_MAX, 12);      // 12位范围
        motor_current_ = uintToFloat(torque_uint, T_MIN, T_MAX, 12); // 12位范围
        motor_temperature_ = static_cast<float>(temp_raw) / 10.0f;

        // 应答指令1没有故障信息，保持原有error_id_
    }

    // 收到反馈，清零计数器（两种格式都需要）
    response_count_ = 0;
    if (rs05_print_count <= 5) {
        printf("[RS05 Motor %d] Pos: %.4f | Vel: %.4f | Torque: %.4f | Temp: %.1f C | Fault: 0x%02X\n", 
                motor_id_, 
                motor_pos_.load(), 
                motor_spd_.load(), 
                motor_current_.load(), 
                motor_temperature_.load(),
                error_id_.load());
    }
    rs05_print_count++;
}


void RS05MotorDriver::MotorLock() {
    // 私有协议：disable命令（通信类型4）
    uint8_t payload[8] = {0};
    // 失能是安全关键的状态切换命令。重复发送，降低单帧丢失后
    // 驱动器继续保持最后一条 MIT/PD 指令的风险。
    for (int i = 0; i < 10; ++i) {
        sendCommand(0x04, master_id_, payload);
        if (i < 9) {
            Timer::ThreadSleepFor(10);
        }
    }
}

void RS05MotorDriver::MotorUnlock() {
    // 私有协议：enable命令（通信类型3）
    uint8_t payload[8] = {0};
    // 使能是状态切换命令，连续发送几次可以降低偶发丢帧导致未使能的概率。
    for (int i = 0; i < 5; ++i) {
        sendCommand(0x03, master_id_, payload);
        Timer::ThreadSleepFor(10);
    }
    Timer::ThreadSleepFor(500);
}

uint8_t RS05MotorDriver::MotorInit() {
    MotorUnlock();
    Timer::ThreadSleepFor(100);
    set_motor_control_mode(0);  // 0 = 运控模式（move_control_mode）
    Timer::ThreadSleepFor(100);
    refresh_motor_status();
    Timer::ThreadSleepFor(100);
    return error_id_;
}

void RS05MotorDriver::MotorDeInit() {
    MotorLock();
    Timer::ThreadSleepFor(100);
}

bool RS05MotorDriver::MotorSetZero() {
    // 私有协议：设置零点流程
    // 根据RS05使用说明书
    
    // 步骤1: 失能电机（安全措施）
    MotorLock();
    Timer::ThreadSleepFor(100);
    
    // 步骤2: 切换到速度模式（模式2）
    set_motor_control_mode(2);  // 速度模式
    Timer::ThreadSleepFor(100);
    
    // 步骤3: 发送归零命令（通信类型6）
    uint8_t payload[8] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    sendCommand(0x06, master_id_, payload);
    Timer::ThreadSleepFor(500);  // 等待归零完成
    
    return true;
}

bool RS05MotorDriver::MotorWriteFlash() {
    uint8_t payload[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    sendCommand(0x16, master_id_, payload);
    return true;
}

void RS05MotorDriver::MotorGetParam(uint8_t param_cmd) {
    // 实现参数读取
    uint8_t payload[8] = {0};
    sendCommand(0x02, master_id_, payload);  // 请求反馈
}

void RS05MotorDriver::MotorPosModeCmd(float pos, float spd, bool ignore_limit) {
    // RS05使用私有协议，主要使用MIT模式，位置模式需要设置参数
    set_motor_control_mode(POS);
    // 实现位置控制命令
}

void RS05MotorDriver::MotorSpdModeCmd(float spd) {
    set_motor_control_mode(SPD);
    // 实现速度控制命令
}

void RS05MotorDriver::MotorMitModeCmd(float f_p, float f_v, float f_kp, float f_kd, float f_t) {
    // 安全检查：过滤 NaN/Inf
    if (!std::isfinite(f_p) || !std::isfinite(f_v) ||
        !std::isfinite(f_kp) || !std::isfinite(f_kd) || !std::isfinite(f_t)) {
        return;
    }
    // 硬限幅，防止超出物理能力
    f_p = std::clamp(f_p, P_MIN, P_MAX);
    f_v = std::clamp(f_v, V_MIN, V_MAX);
    f_kp = std::clamp(f_kp, KP_MIN, KP_MAX);
    f_kd = std::clamp(f_kd, KD_MIN, KD_MAX);
    f_t = std::clamp(f_t, T_MIN, T_MAX);

    // 私有协议：运控模式（通信类型0x01）
    // 数据格式：位置(16位) + 速度(16位) + Kp(16位) + Kd(16位)
    // CAN ID格式：通信类型(5位) | 力矩(16位) | 电机ID(8位)
    uint16_t t_uint = floatToUint(f_t, T_MIN, T_MAX, 16);
    uint16_t p_uint = floatToUint(f_p, P_MIN, P_MAX, 16);
    uint16_t v_uint = floatToUint(f_v, V_MIN, V_MAX, 16);
    uint16_t kp_uint = floatToUint(f_kp, KP_MIN, KP_MAX, 16);
    uint16_t kd_uint = floatToUint(f_kd, KD_MIN, KD_MAX, 16);
    
    uint8_t payload[8];
    payload[0] = (p_uint >> 8) & 0xFF;  // 位置高字节
    payload[1] = p_uint & 0xFF;          // 位置低字节
    payload[2] = (v_uint >> 8) & 0xFF;  // 速度高字节
    payload[3] = v_uint & 0xFF;          // 速度低字节
    payload[4] = (kp_uint >> 8) & 0xFF;  // Kp高字节
    payload[5] = kp_uint & 0xFF;         // Kp低字节
    payload[6] = (kd_uint >> 8) & 0xFF;  // Kd高字节
    payload[7] = kd_uint & 0xFF;         // Kd低字节
    
    // 通信类型0x01（运控模式），数据字段为力矩值
    sendCommand(0x01, t_uint, payload);
}

void RS05MotorDriver::set_motor_control_mode(uint8_t motor_control_mode) {
    motor_control_mode_ = motor_control_mode;
    // 私有协议：设置运行模式（参数0x7005）
    // 0=运控模式, 1=位置模式, 2=速度模式, 3=电流模式, 4=零点模式, 5=位置模式CSP
    uint8_t payload[8] = {0};
    payload[0] = 0x05;  // 参数地址低字节
    payload[1] = 0x70;  // 参数地址高字节
    payload[4] = static_cast<uint8_t>(motor_control_mode);
    sendCommand(0x12, master_id_, payload);  // 通信类型0x12 = 设置参数
}

void RS05MotorDriver::refresh_motor_status() {
    // 私有协议：请求电机反馈（通信类型2）
    uint8_t payload[8] = {0};
    sendCommand(0x02, master_id_, payload);
}

void RS05MotorDriver::clear_motor_error() {
    // 私有协议：disable然后enable
    uint8_t payload[8] = {0};
    sendCommand(0x04, master_id_, payload);
    Timer::ThreadSleepFor(100);
    sendCommand(0x03, master_id_, payload);
}

void RS05MotorDriver::MotorResetID() {
    // 识别电机ID：发送查询命令到广播ID，等待电机回复
    // 电机上电时会发送广播帧 0x7FFE，我们发送查询命令让它回复实际ID
    
    if (logger_) {
        logger_->info("[Motor {}] Querying motor ID from broadcast address...", motor_id_);
    }
    
    // 发送查询命令到广播ID 0x7FFE（私有协议）
    // 通信类型 0x11 (参数读取), 参数地址 0x7001 (电机ID参数)
    uint32_t comm_type_bits = (0x11 & 0x1F) << 24;  // 通信类型 0x11 (读取参数)
    uint32_t data_field_bits = (0x7FFE & 0xFFFF) << 8;  // 数据字段使用广播ID
    uint32_t id_bits = 0xFE & 0xFF;  // 广播ID的低8位是 0xFE
    uint32_t broadcast_can_id = comm_type_bits | data_field_bits | id_bits;
    
    can_frame query_frame{};
    query_frame.can_id = broadcast_can_id | CAN_EFF_FLAG;
    query_frame.can_dlc = 8;
    
    // 参数读取格式: [参数地址低字节, 参数地址高字节, ...]
    query_frame.data[0] = 0x01;  // 参数地址低字节 (0x7001 - 电机ID)
    query_frame.data[1] = 0x70;  // 参数地址高字节 (0x7001)
    std::memset(&query_frame.data[2], 0, 6);
    
    can_->transmit(query_frame);
    Timer::ThreadSleepFor(200);
    
    if (logger_) {
        logger_->info("[Motor {}] ID query command sent to broadcast address 0x7FFE", motor_id_);
        logger_->info("[Motor {}] Waiting for motor response... (check CAN bus for reply frames)", motor_id_);
        logger_->info("[Motor {}] Motor will reply with its actual ID in the CAN frame ID (low 8 bits)", motor_id_);
    }
}

void RS05MotorDriver::MotorReadVersion() {
    // 通信类型 26：版本号读取
    // 设置版本查询标志位
    version_query_pending_.store(true);

    // 发送版本号读取命令
    uint8_t payload[8] = {0};
    sendCommand(0x1A, 0x0000, payload);  // 通信类型 0x1A (26), 数据字段 0x0000

    // 注意：版本号查询的返回帧通信类型仍然是0x02，但在canRxCallback中会特殊处理
}
