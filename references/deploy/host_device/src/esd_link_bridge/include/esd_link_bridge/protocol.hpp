#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace esd_link_bridge::protocol {

constexpr std::uint8_t kMagic = 0xA5;
constexpr std::uint8_t kWireVersion = 1;
constexpr std::size_t kMaxPayload = 480;
constexpr std::size_t kMaxRaw = 496;
constexpr std::size_t kMaxEncoded = 498;
constexpr std::size_t kMaxWire = 500;

enum class Kind : std::uint8_t { Request = 0, Response = 1, Event = 2 };
enum class DecodeError { None, Cobs, Overflow, RawLength, Magic, Version, Kind, Flags, PayloadLength, Crc };

constexpr std::uint8_t kServiceSession = 0x01;
constexpr std::uint8_t kServiceControl = 0x02;
constexpr std::uint8_t kServiceState = 0x03;
constexpr std::uint8_t kServiceConfig = 0x04;
constexpr std::uint8_t kSessionOpen = 0x01;
constexpr std::uint8_t kActuatorCommand = 0x01;
constexpr std::uint8_t kSetEnable = 0x02;
constexpr std::uint8_t kRobotState = 0x01;
constexpr std::uint8_t kDeviceStatus = 0x02;
constexpr std::uint8_t kSetZero = 0x01;

struct Frame {
  Kind kind{Kind::Event};
  std::uint8_t flags{0};
  std::uint8_t service{0};
  std::uint8_t message{0};
  std::uint32_t sequence{0};
  std::array<std::uint8_t, kMaxPayload> payload{};
  std::size_t payload_len{0};
};

struct WireBuffer {
  std::array<std::uint8_t, kMaxWire> bytes{};
  std::size_t size{0};
};

struct PortState {
  std::uint8_t id{0};
  std::uint8_t valid_mask{0};
  float position{0};
  float velocity{0};
  float effort{0};
};

struct RobotState {
  std::uint32_t session_id{0};
  std::uint16_t schema_id{0};
  std::uint32_t sample_time_us{0};
  std::uint32_t state_sample_seq{0};
  std::uint32_t last_applied_command_seq{0};
  std::uint8_t command_status_flags{0};
  std::uint8_t control_state{0};
  std::uint8_t imu_valid_mask{0};
  std::array<float, 3> gyro{};
  std::array<float, 4> quaternion{};
  std::array<float, 3> accel{};
  std::array<PortState, 8> ports{};
  std::size_t port_count{0};
};

struct SessionInfo {
  std::uint16_t status{0};
  std::uint32_t host_nonce{0};
  std::uint32_t lower_boot_id{0};
  std::uint32_t session_id{0};
  std::uint8_t business_version{0};
  std::uint16_t layout_id{0};
  std::uint32_t active_port_mask{0};
  std::uint16_t schema_id{0};
  std::uint32_t config_fingerprint{0};
  std::uint16_t control_rate_hz{0};
  std::uint32_t device_capabilities{0};
  std::uint8_t control_state{0};
};

struct DeviceStatus {
  std::uint32_t session_id{0};
  std::uint32_t lower_boot_id{0};
  std::uint8_t control_state{0};
  std::uint32_t fault_flags{0};
  std::uint16_t last_reject_code{0};
  std::uint32_t valid_command_count{0};
  std::uint32_t invalid_frame_count{0};
  std::uint32_t rejected_command_count{0};
  std::uint16_t command_age_ms{0};
  std::uint32_t active_port_mask{0};
  std::uint32_t offline_port_mask{0};
};

struct TransactionResult {
  std::uint16_t status{0};
  std::uint32_t requested_session_id{0};
  std::uint32_t current_session_id{0};
  std::uint32_t transaction_id{0};
  std::uint8_t failed_port_id{0};
  std::uint16_t detail_code{0};
  std::uint32_t config_fingerprint{0};
  std::uint32_t requested_port_mask{0};
  std::uint32_t affected_port_mask{0};
  std::uint32_t verified_port_mask{0};
  std::uint32_t rollback_port_mask{0};
};

struct ActuatorCommand {
  std::uint8_t port_id{0};
  float position{0};
  float velocity{0};
  float kp{0};
  float kd{0};
  float effort{0};
};

class StreamDecoder {
 public:
  using Callback = std::function<void(const Frame &, std::uint64_t)>;
  explicit StreamDecoder(Callback callback);
  void feed(const std::uint8_t *data, std::size_t size, std::uint64_t receive_time_ns);
  void reset();
  std::uint64_t frames() const { return frames_.load(std::memory_order_relaxed); }
  std::uint64_t cobs_errors() const { return cobs_errors_.load(std::memory_order_relaxed); }
  std::uint64_t crc_errors() const { return crc_errors_.load(std::memory_order_relaxed); }
  DecodeError last_error() const { return last_error_.load(std::memory_order_relaxed); }

 private:
  void finish(std::uint64_t receive_time_ns);
  Callback callback_;
  std::array<std::uint8_t, kMaxEncoded> encoded_{};
  std::size_t encoded_len_{0};
  bool in_frame_{false};
  bool dropping_{false};
  std::atomic<std::uint64_t> frames_{0};
  std::atomic<std::uint64_t> cobs_errors_{0};
  std::atomic<std::uint64_t> crc_errors_{0};
  std::atomic<DecodeError> last_error_{DecodeError::None};
};

std::uint32_t crc32(const std::uint8_t *data, std::size_t size);
DecodeError decode_encoded(const std::uint8_t *encoded, std::size_t size, Frame &out);
bool encode_frame(const Frame &frame, WireBuffer &out);
bool build_open_session(std::uint32_t nonce, std::uint16_t layout_id, std::uint16_t schema_id,
                        std::uint32_t expected_fingerprint, Frame &out);
bool build_set_enable(std::uint32_t session_id, std::uint32_t transaction_id, bool enable,
                      Frame &out);
bool build_set_zero(std::uint32_t session_id, std::uint32_t transaction_id, bool persist,
                    const std::uint8_t *ports, const float *assigned, std::size_t count, Frame &out);
bool build_actuator_command(std::uint32_t session_id, std::uint16_t layout_id,
                            std::uint32_t source_state_sequence, std::uint32_t fingerprint,
                            const std::array<ActuatorCommand, 8> &commands, Frame &out);
bool parse_session_info(const Frame &frame, SessionInfo &out);
bool parse_robot_state(const Frame &frame, RobotState &out);
bool parse_device_status(const Frame &frame, DeviceStatus &out);
bool parse_transaction_result(const Frame &frame, TransactionResult &out);
std::uint16_t response_status(const Frame &frame);
const char *status_name(std::uint16_t status);

}  // namespace esd_link_bridge::protocol
