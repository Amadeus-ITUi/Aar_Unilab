#include "motor_driver.hpp"
#include "utils.hpp"
#include "rs00_motor_driver.hpp"
#include "rs05_motor_driver.hpp"
#include "el05_motor_driver.hpp"

MotorDriver::MotorDriver() {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_st>());
    logger_ = setup_logger(sinks);
}
std::shared_ptr<MotorDriver> MotorDriver::MotorCreate(uint16_t motor_id, const char* interface,
                                                      const std::string motor_type, uint16_t master_id_offset,
                                                      int motor_model) {
    if (motor_type == "RS00") {
        return std::make_shared<RS00MotorDriver>(motor_id, std::string(interface), master_id_offset);
    } else if (motor_type == "RS05") {
        return std::make_shared<RS05MotorDriver>(motor_id, std::string(interface), master_id_offset);
    } else if (motor_type == "EL05") {
        return std::make_shared<EL05MotorDriver>(motor_id, std::string(interface), master_id_offset);
    } else {
        throw std::runtime_error("Motor type not supported: " + motor_type + ". Supported types: RS00, RS05, EL05");
    }
}
