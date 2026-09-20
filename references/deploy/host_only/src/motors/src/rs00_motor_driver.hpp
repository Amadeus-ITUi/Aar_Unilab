#pragma once

#include "motor_driver.hpp"
#include "SocketCAN.hpp"
#include <atomic>
#include <memory>
#include <string>
#include <linux/can.h>

class RS00MotorDriver : public MotorDriver {
public:
    // RS00电机参数限制
    static constexpr float P_MIN = -12.57f;
    static constexpr float P_MAX = 12.57f;
    static constexpr float V_MIN = -50.0f;
    static constexpr float V_MAX = 50.0f;
    static constexpr float KP_MIN = 0.0f;
    static constexpr float KP_MAX = 500.0f;
    static constexpr float KD_MIN = 0.0f;
    static constexpr float KD_MAX = 5.0f;
    static constexpr float T_MIN = -14.0f;
    static constexpr float T_MAX = 14.0f;

    RS00MotorDriver(uint16_t motor_id, const std::string& can_interface, 
                    uint16_t master_id = 0);
    ~RS00MotorDriver();

    virtual void MotorLock() override;
    virtual void MotorUnlock() override;
    virtual uint8_t MotorInit() override;
    virtual void MotorDeInit() override;
    virtual bool MotorSetZero() override;
    virtual bool MotorWriteFlash() override;
    virtual void MotorGetParam(uint8_t param_cmd) override;
    virtual void MotorPosModeCmd(float pos, float spd, bool ignore_limit = false) override;
    virtual void MotorSpdModeCmd(float spd) override;
    virtual void MotorMitModeCmd(float f_p, float f_v, float f_kp, float f_kd, float f_t) override;
    virtual void MotorResetID() override;
    virtual void set_motor_control_mode(uint8_t motor_control_mode) override;
    virtual int get_response_count() const override { return response_count_; }
    virtual std::uint64_t get_feedback_count() const override {
        return static_cast<std::uint64_t>(frame_counter_.load());
    }
    virtual void refresh_motor_status() override;
    virtual void clear_motor_error() override;

private:
    std::atomic<int> response_count_{0};
    std::atomic<int> frame_counter_{0};  // 帧计数器，用于每100帧打印一次
    std::shared_ptr<SocketCAN> can_;
    std::string can_interface_;
    uint16_t master_id_;

    // 辅助函数
    uint32_t buildCanID(uint8_t comm_type, uint16_t data_field) const;
    uint16_t floatToUint(float x, float x_min, float x_max, int bits) const;
    float uintToFloat(uint16_t x_int, float x_min, float x_max, int bits) const;
    // 参考代码的转换公式：((value / 32767.0f) - 1.0f) * range
    float referenceCodeConvert(uint16_t value, float range) const;
    void sendCommand(uint8_t comm_type, uint16_t data_field, const uint8_t* payload);
    void canRxCallback(const can_frame& rx_frame);
};
