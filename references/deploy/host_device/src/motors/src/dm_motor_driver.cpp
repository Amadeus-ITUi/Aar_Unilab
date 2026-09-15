// DmMotorDriver.cpp
#include "dm_motor_driver.hpp"
#include "timer.hpp"
#include "utils.hpp"
#include <cstring>
#include <map>
#include <cmath>

DM_Limit_Param limit_param[Num_Of_Motor] = {
    {12.5, 20, 28, 500, 5},   // DM4340P_48V
    {12.5, 25, 200, 500, 5},  // DM10010L_48V
    {12.5, 50, 10, 500, 5},   // DM3507
    {12.5, 50, 1.0, 500, 5},  // DM3510
};

DmMotorDriver::DmMotorDriver(uint16_t motor_id, const std::string& interface_type, const std::string& can_interface, uint16_t master_id_offset,
                             DM_Motor_Model motor_model)
    : MotorDriver(), can_(SocketCAN::get(can_interface)), motor_model_(motor_model) {
    if (interface_type != "can") {
        throw std::runtime_error("DM driver only support CAN interface");
    }
    motor_id_ = motor_id;
    // DM 协议里，参数/状态反馈使用的标准 CAN ID 就是 master_id。
    // 约定规则：每个电机的 master_id = 从站ID(slave_id) + 10
    // 例如：电机1 -> master_id=11, 电机2 -> master_id=12, 电机3 -> master_id=13, ...
    // 如果传入的 master_id_offset > 0x7F（如 0xFD），说明是旧的配置格式，需要转换：
    // 旧格式：0xFD -> 0x7D (两者相差 0x80)
    // 新格式：直接使用传入的值（11-16）
    if (master_id_offset > 0x7F) {
        master_id_ = static_cast<uint16_t>(master_id_offset - 0x80);  // 0xFD -> 0x7D (兼容旧配置)
    } else {
        master_id_ = master_id_offset;  // 新格式：直接使用 (11-16)
    }
    limit_param_ = limit_param[motor_model_];
    can_interface_ = can_interface;
    CanCbkFunc can_callback = std::bind(&DmMotorDriver::can_rx_cbk, this, std::placeholders::_1);
    can_->add_can_callback(can_callback, master_id_, false);  // 标准帧，使用master_id作为回调ID
}

DmMotorDriver::~DmMotorDriver() { can_->remove_can_callback(master_id_); }

void DmMotorDriver::MotorLock() {
    can_frame tx_frame;
    tx_frame.can_id = motor_id_;  // change according to the mode
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = 0xFF;
    tx_frame.data[1] = 0xFF;
    tx_frame.data[2] = 0xFF;
    tx_frame.data[3] = 0xFF;
    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    // 对应 Damiao 协议：0xFD = Disable（失能）
    tx_frame.data[7] = 0xFD;

    // 多次发送失能命令，确保电机收到（安全措施）
    // 失能是安全关键操作，必须确保成功
    bool transmitted = false;
    for (int i = 0; i < 5; ++i) {
        transmitted = can_->transmit(tx_frame) || transmitted;
        if (i < 4) {  // 最后一次不需要等待
            Timer::ThreadSleepFor(20);  // 每次间隔20ms
        }
    }
    if (transmitted) {
        response_count_++;
    }
}

void DmMotorDriver::MotorUnlock() {
    can_frame tx_frame;
    tx_frame.can_id = motor_id_;  // change according to the mode
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = 0xFF;
    tx_frame.data[1] = 0xFF;
    tx_frame.data[2] = 0xFF;
    tx_frame.data[3] = 0xFF;
    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    // 对应 Damiao 协议：0xFC = Enable（使能）
    tx_frame.data[7] = 0xFC;
    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

uint8_t DmMotorDriver::MotorInit() {
    // send disable command to enter read mode
    DmMotorDriver::MotorUnlock();
    Timer::ThreadSleepFor(normal_sleep_time);
    set_motor_control_mode(MIT);
    Timer::ThreadSleepFor(normal_sleep_time);
    // send enable command to enter contorl mode
    DmMotorDriver::MotorLock();
    Timer::ThreadSleepFor(normal_sleep_time);
    DmMotorDriver::refresh_motor_status();
    Timer::ThreadSleepFor(normal_sleep_time);
    switch (error_id_) {
        case DMError::DM_DOWN:
            return DMError::DM_DOWN;
            break;
        case DMError::DM_UP:
            return DMError::DM_UP;
            break;
        case DMError::LOST_CONN:
            return DMError::LOST_CONN;
            break;
        case DMError::OVER_CURRENT:
            return DMError::OVER_CURRENT;
            break;
        case DMError::MOS_OVER_TEMP:
            return DMError::MOS_OVER_TEMP;
            break;
        case DMError::COIL_OVER_TEMP:
            return DMError::COIL_OVER_TEMP;
            break;
        case DMError::UNDER_VOLT:
            return DMError::UNDER_VOLT;
            break;
        case DMError::OVER_VOLT:
            return DMError::OVER_VOLT;
            break;
        case DMError::OVER_LOAD:
            return DMError::OVER_LOAD;
            break;
        default:
            return error_id_;
    }
    return error_id_;
}

void DmMotorDriver::MotorDeInit() {
    // 反初始化时确保电机失能（安全措施）
    // 注意：不要在反初始化时使能电机，这会导致节点关闭后电机仍然运行
    DmMotorDriver::MotorLock();  // 改为失能，而不是使能
    Timer::ThreadSleepFor(normal_sleep_time);
}

bool DmMotorDriver::MotorSetZero() {
    // send set zero command
    DmMotorDriver::set_motor_zero_dm();
    Timer::ThreadSleepFor(setup_sleep_time);  // wait for motor to set zero
    logger_->info("motor_id: {0}\tposition: {1}\t", motor_id_, get_motor_pos());
    DmMotorDriver::MotorUnlock();
    if (get_motor_pos() > judgment_accuracy_threshold || get_motor_pos() < -judgment_accuracy_threshold) {
        logger_->warn("set zero error");
        return false;
    } else {
        logger_->info("set zero success");
        return true;
    }
    // disable motor
}

void DmMotorDriver::can_rx_cbk(const can_frame& rx_frame) {
    // 首先检查是否是参数反馈（通过master_id接收，data[2]==0x33）
    // 参数反馈格式：CAN ID = master_id (0xFD=0x07D), data[0-1]=电机ID, data[2]=0x33, data[3]=寄存器ID
    // 从candump可以看到：07D  03 00 33 08 03 00 00 00
    // CAN ID = 0x07D (master_id), data[0]=0x03 (电机ID), data[1]=0x00, data[2]=0x33, data[3]=0x08 (寄存器ID)
    if (rx_frame.can_id == master_id_ && rx_frame.data[2] == 0x33) {
        // 提取电机ID（低16位，小端序：data[0]是低字节，data[1]是高字节）
        uint16_t slave_id = (static_cast<uint16_t>(rx_frame.data[1]) << 8) | rx_frame.data[0];
        
        // 检查是否是本电机的参数反馈
        if (slave_id == motor_id_) {
            uint8_t rid = rx_frame.data[3];
            
            std::lock_guard<std::mutex> lock(param_map_mutex_);
            ParamValue pv;
            pv.timestamp = std::chrono::steady_clock::now();
            
            if (is_in_ranges(rid)) {
                // uint32类型（小端序）
                uint32_t data_uint32 = (static_cast<uint32_t>(rx_frame.data[7]) << 24) |
                                      (static_cast<uint32_t>(rx_frame.data[6]) << 16) |
                                      (static_cast<uint32_t>(rx_frame.data[5]) << 8) |
                                      rx_frame.data[4];
                pv.value.uint32Value = data_uint32;
                pv.isFloat = false;
            } else {
                // float类型（小端序）
                pv.value.floatValue = uint8_to_float(rx_frame.data + 4);
                pv.isFloat = true;
            }
            
            param_map_[rid] = pv;
            
            // 收到参数反馈，也清零response_count_
            {
                response_count_ = 0;
            }
            
            // 调试日志（仅在检测阶段输出）
            // static std::map<uint8_t, int> debug_count;
            // if (++debug_count[rid] <= 3) {
            //     if (logger_) {
            //         if (is_in_ranges(rid)) {
            //             logger_->info("收到参数反馈: motor_id={}, RID={}, value={}", 
            //                          motor_id_, rid, pv.value.uint32Value);
            //         } else {
            //             logger_->info("收到参数反馈: motor_id={}, RID={}, value={:.2f}", 
            //                          motor_id_, rid, pv.value.floatValue);
            //         }
            //     }
            // }
            
            return;  // 参数反馈处理完成，直接返回
        }
    }
    
    // 处理电机状态反馈（MIT模式反馈）
    // 状态反馈格式：CAN ID = master_id, data[0]低4位可能包含电机ID（用于多电机共享master_id的情况）
    if (rx_frame.can_id == master_id_) {
        // 检查data[0]的低4位是否匹配本电机ID（用于区分共享master_id的多个电机）
        uint8_t slave_id_from_data = rx_frame.data[0] & 0x0F;
        if (slave_id_from_data != motor_id_ && slave_id_from_data != 0) {
            // 如果data[0]中的ID不匹配且不为0，这个反馈不属于本电机
            return;
        }
        
        {
            response_count_ = 0;
        }
        uint16_t master_id_t = 0;
        uint16_t pos_int = 0;
        uint16_t spd_int = 0;
        uint16_t t_int = 0;
        pos_int = rx_frame.data[1] << 8 | rx_frame.data[2];
        spd_int = rx_frame.data[3] << 4 | (rx_frame.data[4] & 0xF0) >> 4;
        t_int = (rx_frame.data[4] & 0x0F) << 8 | rx_frame.data[5];
        master_id_t = rx_frame.can_id;
        if ((rx_frame.data[0] & 0xF0) >> 4 > 7) {  // error code range from 8 to 15
            error_id_ = (rx_frame.data[0] & 0xF0) >> 4;
            if (logger_) {
                logger_->error("can_interface: {0}\tmotor_id: {1}\terror_id: 0x{2:x}", can_interface_, motor_id_, (uint32_t)error_id_);
            }
        }
        motor_pos_ =
            range_map(pos_int, uint16_t(0), bitmax<uint16_t>(16), -limit_param_.PosMax, limit_param_.PosMax);
        motor_spd_ =
            range_map(spd_int, uint16_t(0), bitmax<uint16_t>(12), -limit_param_.SpdMax, limit_param_.SpdMax);
        motor_current_ =
            range_map(t_int, uint16_t(0), bitmax<uint16_t>(12), -limit_param_.TauMax, limit_param_.TauMax);
        mos_temperature_ = rx_frame.data[6];
        motor_temperature_ = rx_frame.data[7];
    }
}

void DmMotorDriver::MotorGetParam(uint8_t param_cmd) {
    can_frame tx_frame;
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = motor_id_ & 0xFF;
    tx_frame.data[1] = motor_id_ >> 8;
    tx_frame.data[2] = 0x33;
    tx_frame.data[3] = param_cmd;

    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    tx_frame.data[7] = 0xFF;
    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

void DmMotorDriver::MotorPosModeCmd(float pos, float spd, bool ignore_limit) {
    if (motor_control_mode_ != POS) {
        set_motor_control_mode(POS);
        return;
    }
    can_frame tx_frame;
    tx_frame.can_id = 0x100 + motor_id_;
    tx_frame.can_dlc = 0x08;
    uint8_t *pbuf, *vbuf;

    spd = limit(spd, -limit_param_.SpdMax, limit_param_.SpdMax);
    pos = limit(pos, -limit_param_.PosMax, limit_param_.PosMax);

    pbuf = (uint8_t*)&pos;
    vbuf = (uint8_t*)&spd;

    tx_frame.data[0] = *pbuf;
    tx_frame.data[1] = *(pbuf + 1);
    tx_frame.data[2] = *(pbuf + 2);
    tx_frame.data[3] = *(pbuf + 3);
    tx_frame.data[4] = *vbuf;
    tx_frame.data[5] = *(vbuf + 1);
    tx_frame.data[6] = *(vbuf + 2);
    tx_frame.data[7] = *(vbuf + 3);

    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

void DmMotorDriver::MotorSpdModeCmd(float spd) {
    if (motor_control_mode_ != SPD) {
        set_motor_control_mode(SPD);
        return;
    }
    can_frame tx_frame;
    tx_frame.can_id = 0x200 + motor_id_;
    tx_frame.can_dlc = 0x04;

    spd = limit(spd, -limit_param_.SpdMax, limit_param_.SpdMax);
    union32_t rv_type_convert;
    rv_type_convert.f = spd;
    tx_frame.data[0] = rv_type_convert.buf[0];
    tx_frame.data[1] = rv_type_convert.buf[1];
    tx_frame.data[2] = rv_type_convert.buf[2];
    tx_frame.data[3] = rv_type_convert.buf[3];

    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

// Transmit MIT-mDme control(hybrid) package. Called in canTask.
void DmMotorDriver::MotorMitModeCmd(float f_p, float f_v, float f_kp, float f_kd, float f_t) {
    if (motor_control_mode_ != MIT) {
        set_motor_control_mode(MIT);
        return;
    }
    uint16_t p, v, kp, kd, t;
    can_frame tx_frame;

    f_p = limit(f_p, -limit_param_.PosMax, limit_param_.PosMax);
    f_v = limit(f_v, -limit_param_.SpdMax, limit_param_.SpdMax);
    f_kp = limit(f_kp, 0.0f, limit_param_.OKpMax);
    f_kd = limit(f_kd, 0.0f, limit_param_.OKdMax);
    f_t = limit(f_t, -limit_param_.TauMax, limit_param_.TauMax);

    p = range_map(f_p, -limit_param_.PosMax, limit_param_.PosMax, uint16_t(0), bitmax<uint16_t>(16));
    v = range_map(f_v, -limit_param_.SpdMax, limit_param_.SpdMax, uint16_t(0), bitmax<uint16_t>(12));
    kp = range_map(f_kp, 0.0f, limit_param_.OKpMax, uint16_t(0), bitmax<uint16_t>(12));
    kd = range_map(f_kd, 0.0f, limit_param_.OKdMax, uint16_t(0), bitmax<uint16_t>(12));
    t = range_map(f_t, -limit_param_.TauMax, limit_param_.TauMax, uint16_t(0), bitmax<uint16_t>(12));

    tx_frame.can_id = motor_id_;
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = p >> 8;
    tx_frame.data[1] = p & 0xFF;
    tx_frame.data[2] = v >> 4;
    tx_frame.data[3] = (v & 0x0F) << 4 | kp >> 8;
    tx_frame.data[4] = kp & 0xFF;
    tx_frame.data[5] = kd >> 4;
    tx_frame.data[6] = (kd & 0x0F) << 4 | t >> 8;
    tx_frame.data[7] = t & 0xFF;

    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

void DmMotorDriver::set_motor_control_mode(uint8_t motor_control_mode) {
    write_register_dm(10, motor_control_mode);
    motor_control_mode_ = motor_control_mode;
}

void DmMotorDriver::set_motor_zero_dm() {
    can_frame tx_frame;
    tx_frame.can_id = motor_id_;  // change according to the mode
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = 0xFF;
    tx_frame.data[1] = 0xFF;
    tx_frame.data[2] = 0xFF;
    tx_frame.data[3] = 0xFF;
    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    tx_frame.data[7] = 0xFE;
    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

void DmMotorDriver::clear_motor_error_dm() {
    can_frame tx_frame;
    tx_frame.can_id = motor_id_;  // change according to the mode
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = 0xFF;
    tx_frame.data[1] = 0xFF;
    tx_frame.data[2] = 0xFF;
    tx_frame.data[3] = 0xFF;
    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    tx_frame.data[7] = 0xFB;
    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

void DmMotorDriver::write_register_dm(uint8_t rid, float value) {
    param_cmd_flag_[rid] = false;
    can_frame tx_frame;
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x08;

    uint8_t* vbuf;
    vbuf = (uint8_t*)&value;

    tx_frame.data[0] = motor_id_ & 0xFF;
    tx_frame.data[1] = motor_id_ >> 8;
    tx_frame.data[2] = 0x55;
    tx_frame.data[3] = rid;

    tx_frame.data[4] = *vbuf;
    tx_frame.data[5] = *(vbuf + 1);
    tx_frame.data[6] = *(vbuf + 2);
    tx_frame.data[7] = *(vbuf + 3);
    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

void DmMotorDriver::write_register_dm(uint8_t rid, int32_t value) {
    param_cmd_flag_[rid] = false;
    can_frame tx_frame;
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x08;

    uint8_t* vbuf;
    vbuf = (uint8_t*)&value;

    tx_frame.data[0] = motor_id_ & 0xFF;
    tx_frame.data[1] = motor_id_ >> 8;
    tx_frame.data[2] = 0x55;
    tx_frame.data[3] = rid;

    tx_frame.data[4] = *vbuf;
    tx_frame.data[5] = *(vbuf + 1);
    tx_frame.data[6] = *(vbuf + 2);
    tx_frame.data[7] = *(vbuf + 3);
    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

void DmMotorDriver::save_register_dm(uint8_t rid) {
    can_frame tx_frame;
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = motor_id_ & 0xFF;
    tx_frame.data[1] = motor_id_ >> 8;
    tx_frame.data[2] = 0xAA;
    tx_frame.data[3] = rid;

    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    tx_frame.data[7] = 0xFF;
    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

void DmMotorDriver::refresh_motor_status() {
    can_frame tx_frame;
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = motor_id_ & 0xFF;
    tx_frame.data[1] = motor_id_ >> 8;
    tx_frame.data[2] = 0xCC;
    tx_frame.data[3] = 0x00;

    tx_frame.data[4] = 0x00;
    tx_frame.data[5] = 0x00;
    tx_frame.data[6] = 0x00;
    tx_frame.data[7] = 0x00;
    if (can_->transmit(tx_frame)) {
        response_count_++;
    }
}

void DmMotorDriver::clear_motor_error() {
    clear_motor_error_dm();
}

bool DmMotorDriver::MotorWriteFlash() {
    return true;
}

// 写入 master_id 并保存到 flash
bool DmMotorDriver::set_master_id(uint16_t new_master_id) {
    uint16_t old_master_id = master_id_;  // 保存旧的 master_id，用于验证时匹配反馈帧
    
    if (logger_) {
        logger_->info("写入 master_id: motor_id={}, old_master_id={}, new_master_id={}", 
                     motor_id_, old_master_id, new_master_id);
    }
    
    // 1. 多次写入 MST_ID 寄存器（寄存器号 7，uint32 类型），确保写入成功
    // 注意：MST_ID 是 uint32 类型，所以使用 int32_t 版本的 write_register_dm
    if (logger_) {
        logger_->info("步骤1: 写入 MST_ID 寄存器到 RAM...");
    }
    for (int i = 0; i < 3; ++i) {
        write_register_dm(static_cast<uint8_t>(DM_REG::MST_ID), static_cast<int32_t>(new_master_id));
        Timer::ThreadSleepFor(100);  // 每次写入后等待
    }
    
    // 2. 验证写入到RAM（保存到flash前先确认RAM写入成功）
    // 注意：写入后，电机可能立即使用新的 master_id 来发送反馈，也可能还在用旧的
    // 所以我们需要尝试两种 master_id 来匹配反馈帧
    if (logger_) {
        logger_->info("步骤2: 验证 RAM 写入（尝试使用旧 master_id={} 和新 master_id={} 匹配反馈）...", 
                     old_master_id, new_master_id);
    }
    
    // 先尝试用旧的 master_id 读取（因为可能还没生效）
    float read_back_ram = read_motor_param_with_timeout(static_cast<uint8_t>(DM_REG::MST_ID), 800, old_master_id);
    
    // 如果读取失败或值不匹配，尝试用新的 master_id 读取
    if (read_back_ram == 0.0f || std::abs(read_back_ram - static_cast<float>(new_master_id)) >= 0.5f) {
        if (logger_) {
            logger_->info("使用旧 master_id={} 验证失败，尝试使用新 master_id={} 读取...", 
                         old_master_id, new_master_id);
        }
        Timer::ThreadSleepFor(100);  // 等待一下，让电机使用新的 master_id
        read_back_ram = read_motor_param_with_timeout(static_cast<uint8_t>(DM_REG::MST_ID), 800, new_master_id);
    }
    
    if (logger_) {
        logger_->info("RAM 验证结果: expected={}, read_back={}", new_master_id, read_back_ram);
    }
    
    if (read_back_ram == 0.0f) {
        if (logger_) {
            logger_->warn("master_id RAM写入验证失败: 读取超时或未收到反馈 (read_back=0)");
        }
        return false;
    }
    
    if (std::abs(read_back_ram - static_cast<float>(new_master_id)) >= 0.5f) {
        if (logger_) {
            logger_->warn("master_id RAM写入验证失败: motor_id={}, expected={}, read_back={}", 
                         motor_id_, new_master_id, read_back_ram);
        }
        return false;
    }
    
    if (logger_) {
        logger_->info("✓ master_id RAM写入成功: motor_id={}, new_master_id={}", 
                     motor_id_, new_master_id);
    }
    
    // 3. 保存到 flash（多次保存，确保成功）
    if (logger_) {
        logger_->info("步骤3: 保存到 flash...");
    }
    for (int i = 0; i < 3; ++i) {
        save_register_dm(static_cast<uint8_t>(DM_REG::MST_ID));
        Timer::ThreadSleepFor(200);  // flash写入需要更长时间
    }
    
    // 4. 等待flash写入完成（flash写入可能需要更长时间）
    Timer::ThreadSleepFor(1000);  // 额外等待1秒，确保flash写入完成
    
    // 5. 验证写入结果（从flash读取）
    // 注意：写入后，电机可能使用新的 master_id 来发送反馈，也可能还在用旧的
    // 所以我们需要尝试两种 master_id 来匹配反馈帧
    if (logger_) {
        logger_->info("步骤4: 验证 flash 写入（尝试使用旧 master_id={} 和新 master_id={} 匹配反馈）...", 
                     old_master_id, new_master_id);
    }
    
    // 先尝试用旧的 master_id 读取（因为可能还没生效）
    float read_back = read_motor_param_with_timeout(static_cast<uint8_t>(DM_REG::MST_ID), 800, old_master_id);
    
    // 如果读取失败或值不匹配，尝试用新的 master_id 读取
    if (read_back == 0.0f || std::abs(read_back - static_cast<float>(new_master_id)) >= 0.5f) {
        if (logger_) {
            logger_->info("使用旧 master_id={} 验证失败，尝试使用新 master_id={} 读取...", 
                         old_master_id, new_master_id);
        }
        Timer::ThreadSleepFor(200);  // 等待一下，让电机使用新的 master_id
        read_back = read_motor_param_with_timeout(static_cast<uint8_t>(DM_REG::MST_ID), 800, new_master_id);
    }
    
    if (logger_) {
        logger_->info("Flash 验证结果: expected={}, read_back={}", new_master_id, read_back);
    }
    
    if (read_back == 0.0f) {
        if (logger_) {
            logger_->warn("master_id flash写入验证失败: 读取超时或未收到反馈 (read_back=0)");
        }
        return false;
    }
    
    if (std::abs(read_back - static_cast<float>(new_master_id)) < 0.5f) {
        // 更新内部 master_id_
        master_id_ = new_master_id;
        
        // 注意：master_id 改变后，需要更新 SocketCAN 的回调注册
        // 移除旧的回调，添加新的回调
        can_->remove_can_callback(old_master_id);
        CanCbkFunc can_callback = std::bind(&DmMotorDriver::can_rx_cbk, this, std::placeholders::_1);
        can_->add_can_callback(can_callback, new_master_id, false);
        
        if (logger_) {
            logger_->info("✓ master_id 写入成功: motor_id={}, new_master_id={}, 已保存到 flash，回调已更新", 
                         motor_id_, new_master_id);
        }
        return true;
    } else {
        if (logger_) {
            logger_->warn("master_id flash写入验证失败: motor_id={}, expected={}, read_back={}", 
                         motor_id_, new_master_id, read_back);
        }
        return false;
    }
}

// 辅助函数：判断寄存器是否为uint32类型
bool DmMotorDriver::is_in_ranges(int number) {
    return (7 <= number && number <= 10) ||
           (13 <= number && number <= 16) ||
           (35 <= number && number <= 36);
}

float DmMotorDriver::uint8_to_float(const uint8_t data[4]) {
    uint32_t combined = (static_cast<uint32_t>(data[3]) << 24) |
                        (static_cast<uint32_t>(data[2]) << 16) |
                        (static_cast<uint32_t>(data[1]) << 8)  |
                        static_cast<uint32_t>(data[0]);
    float result;
    memcpy(&result, &combined, sizeof(result));
    return result;
}

// 读取电机参数（带等待和超时）
// 注意：这里不依赖 SocketCAN 的回调机制，而是临时开启“嗅探模式”直接抓总线上的帧，
// 这样不会受到 master_id 共享导致回调被覆盖的影响。
float DmMotorDriver::read_motor_param_with_timeout(uint8_t param_cmd, int timeout_ms) {
    // 使用当前的 master_id_ 调用重载版本
    return read_motor_param_with_timeout(param_cmd, timeout_ms, master_id_);
}

// 读取电机参数（带等待和超时，支持指定 master_id）
// 用于在 master_id 改变后验证时使用新的 master_id 匹配反馈帧
float DmMotorDriver::read_motor_param_with_timeout(uint8_t param_cmd, int timeout_ms, uint16_t use_master_id) {
    // 策略：使用回调机制获取参数反馈（更可靠）
    // 因为回调机制已经能正常收到反馈，而嗅探模式可能有时序问题
    
    // 记录开始时间（用于判断参数反馈是否是新发送的）
    auto request_start_time = std::chrono::steady_clock::now();
    
    // 清空之前的参数缓存（避免使用旧的反馈）
    {
        std::lock_guard<std::mutex> lock(param_map_mutex_);
        param_map_.erase(param_cmd);
    }
    
    // 发送参数读取命令（多次发送，确保电机收到）
    for (int i = 0; i < 3; ++i) {
        MotorGetParam(param_cmd);
        Timer::ThreadSleepFor(20);  // 每次发送间隔20ms
    }
    
    // 等待参数反馈（通过回调机制接收）
    auto start_time = std::chrono::steady_clock::now();
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 每10ms检查一次
        
        // 检查是否收到参数反馈
        {
            std::lock_guard<std::mutex> lock(param_map_mutex_);
            auto it = param_map_.find(param_cmd);
            if (it != param_map_.end()) {
                // 检查参数反馈的时间戳，确保是新发送命令的反馈（而不是旧的）
                // 如果参数反馈的时间戳在请求开始时间之后，说明是新反馈
                if (it->second.timestamp >= request_start_time) {
                    // 找到了参数反馈
                    float result = 0.0f;
                    if (is_in_ranges(param_cmd)) {
                        result = static_cast<float>(it->second.value.uint32Value);
                    } else {
                        result = it->second.value.floatValue;
                    }
                    
                    // if (logger_) {
                    //     logger_->info("read_motor_param_with_timeout: 成功读取参数 param_cmd={}, value={} (使用回调机制, use_master_id={})", 
                    //                   param_cmd, result, use_master_id);
                    // }
                    return result;
                }
                // 参数反馈是旧的，继续等待
                // if (logger_) {
                //     logger_->debug("read_motor_param_with_timeout: 收到旧参数反馈，继续等待...");
                // }
            }
        }
        
        // 检查超时
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= timeout_ms) {
            break;
        }
    }
    
    // if (logger_) {
    //     logger_->warn("read_motor_param_with_timeout: 回调机制超时，未收到参数反馈 param_cmd={}, use_master_id={}, motor_id={}", 
    //                   param_cmd, use_master_id, motor_id_);
    // }
    
    // 如果回调机制没有收到反馈，尝试使用嗅探模式作为备选方案
    // if (logger_) {
    //     logger_->debug("read_motor_param_with_timeout: 回调机制未收到反馈，尝试使用嗅探模式...");
    // }
    
    // 步骤1：先开启嗅探模式，捕获可能已经存在的反馈帧
    auto pre_frames = can_->receive_all_frames(50);  // 等待50ms，捕获已有帧
    
    // 步骤2：再次发送参数读取命令
    for (int i = 0; i < 3; ++i) {
        MotorGetParam(param_cmd);
        Timer::ThreadSleepFor(10);  // 每次发送间隔10ms
    }
    
    // 步骤3：立即开启嗅探模式并等待反馈
    auto frames = can_->receive_all_frames(timeout_ms);
    
    // 合并两次捕获的帧
    frames.insert(frames.end(), pre_frames.begin(), pre_frames.end());
    
    // if (logger_) {
    //     logger_->info("read_motor_param_with_timeout: 发送参数读取命令 param_cmd={}, use_master_id={}, 收到 {} 帧 (pre_frames={}, new_frames={})", 
    //                   param_cmd, use_master_id, frames.size(), pre_frames.size(), frames.size() - pre_frames.size());
    // }
    
    // 首先尝试使用指定的 master_id 匹配
    for (const auto &f : frames) {
        bool is_extended = (f.can_id & CAN_EFF_FLAG) != 0;
        uint32_t can_id = is_extended ? (f.can_id & CAN_EFF_MASK) : (f.can_id & CAN_SFF_MASK);

        // 使用指定的 master_id 来匹配反馈帧
        if (!is_extended && can_id == use_master_id) {
            const uint8_t *d = f.data;

            // 参数反馈格式：data[0-1]=SlaveID(小端), data[2]=0x33, data[3]=RID
            uint16_t slave_id = (static_cast<uint16_t>(d[1]) << 8) | d[0];
            uint8_t rid = d[3];

            // if (logger_) {
            //     logger_->debug("read_motor_param_with_timeout: 检查帧 can_id=0x{:X}, slave_id={}, d[2]=0x{:02X}, rid={}, param_cmd={}", 
            //                   can_id, slave_id, d[2], rid, param_cmd);
            // }

            if (d[2] == 0x33 && slave_id == motor_id_ && rid == param_cmd) {
                // 找到了本电机、本寄存器的反馈
                if (is_in_ranges(rid)) {
                    uint32_t data_uint32 =
                        (static_cast<uint32_t>(d[7]) << 24) |
                        (static_cast<uint32_t>(d[6]) << 16) |
                        (static_cast<uint32_t>(d[5]) << 8)  |
                        d[4];
                    float result = static_cast<float>(data_uint32);
                    // if (logger_) {
                    //     logger_->debug("read_motor_param_with_timeout: 成功读取参数 param_cmd={}, value={} (使用指定 master_id={})", 
                    //                   param_cmd, result, use_master_id);
                    // }
                    return result;
                } else {
                    float result = uint8_to_float(d + 4);
                    // if (logger_) {
                    //     logger_->debug("read_motor_param_with_timeout: 成功读取参数 param_cmd={}, value={} (使用指定 master_id={})", 
                    //                   param_cmd, result, use_master_id);
                    // }
                    return result;
                }
            }
        }
    }
    
    // 如果使用指定的 master_id 匹配失败，尝试扫描所有帧（可能 master_id 已经改变）
    // if (logger_) {
    //     logger_->info("read_motor_param_with_timeout: 使用指定 master_id={} 匹配失败，尝试扫描所有帧...", use_master_id);
    // }
    
    for (const auto &f : frames) {
        bool is_extended = (f.can_id & CAN_EFF_FLAG) != 0;
        uint32_t can_id = is_extended ? (f.can_id & CAN_EFF_MASK) : (f.can_id & CAN_SFF_MASK);

        // 扫描所有标准帧，查找匹配的 slave_id 和 rid
        if (!is_extended) {
            const uint8_t *d = f.data;

            // 参数反馈格式：data[0-1]=SlaveID(小端), data[2]=0x33, data[3]=RID
            uint16_t slave_id = (static_cast<uint16_t>(d[1]) << 8) | d[0];
            uint8_t rid = d[3];

            // if (logger_) {
            //     logger_->debug("read_motor_param_with_timeout: 扫描帧 can_id=0x{:X}, slave_id={}, d[2]=0x{:02X}, rid={}, param_cmd={}, motor_id={}", 
            //                   can_id, slave_id, d[2], rid, param_cmd, motor_id_);
            // }

            if (d[2] == 0x33 && slave_id == motor_id_ && rid == param_cmd) {
                // 找到了本电机、本寄存器的反馈（但使用了不同的 master_id）
                // 更新内部的 master_id_，因为电机已经改变了 master_id
                if (can_id != master_id_) {
                    // if (logger_) {
                    //     logger_->info("read_motor_param_with_timeout: 检测到电机 {} 的 master_id 已改变: 旧值={}, 新值={}", 
                    //                   motor_id_, master_id_, can_id);
                    // }
                    // 更新 master_id_ 和回调注册
                    uint16_t old_master_id = master_id_;
                    master_id_ = static_cast<uint16_t>(can_id);
                    can_->remove_can_callback(old_master_id);
                    CanCbkFunc can_callback = std::bind(&DmMotorDriver::can_rx_cbk, this, std::placeholders::_1);
                    can_->add_can_callback(can_callback, master_id_, false);
                }
                
                // 找到了本电机、本寄存器的反馈
                if (is_in_ranges(rid)) {
                    uint32_t data_uint32 =
                        (static_cast<uint32_t>(d[7]) << 24) |
                        (static_cast<uint32_t>(d[6]) << 16) |
                        (static_cast<uint32_t>(d[5]) << 8)  |
                        d[4];
                    float result = static_cast<float>(data_uint32);
                    // if (logger_) {
                    //     logger_->info("read_motor_param_with_timeout: 成功读取参数 param_cmd={}, value={} (使用扫描到的 master_id={})", 
                    //                   param_cmd, result, can_id);
                    // }
                    return result;
                } else {
                    float result = uint8_to_float(d + 4);
                    // if (logger_) {
                    //     logger_->info("read_motor_param_with_timeout: 成功读取参数 param_cmd={}, value={} (使用扫描到的 master_id={})", 
                    //                   param_cmd, result, can_id);
                    // }
                    return result;
                }
            }
        }
    }

    // 未在超时时间内收到有效反馈，返回0表示读取失败
    // if (logger_) {
    //     logger_->warn("read_motor_param_with_timeout: 未找到匹配的参数反馈 param_cmd={}, use_master_id={}, motor_id={}", 
    //                   param_cmd, use_master_id, motor_id_);
    // }
    return 0.0f;
}
