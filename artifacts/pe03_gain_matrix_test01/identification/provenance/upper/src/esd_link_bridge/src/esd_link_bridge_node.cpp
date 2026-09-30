#include "esd_link_bridge/protocol.hpp"
#include "esd_link_bridge/command_arbiter.hpp"
#include "esd_link_bridge/latest_mailbox.hpp"
#include "esd_link_bridge/coordinates.hpp"
#include "esd_link_bridge/robot_profile.hpp"
#include "esd_link_bridge/robot_state_mapper.hpp"
#include "esd_link_bridge/serial_device.hpp"
#include "esd_link_bridge/time_utils.hpp"
#include "esd_link_bridge/wing_rc_input.hpp"

#include <esd_link_msgs/msg/link_status.hpp>
#include <esd_link_msgs/msg/lower_state.hpp>
#include <esd_link_msgs/msg/actuator_command.hpp>
#include <esd_link_msgs/msg/robot_state.hpp>
#include <esd_link_msgs/srv/set_control_mode.hpp>
#include <esd_link_msgs/srv/set_lower_zeros.hpp>
#include <motors/msg/motor_runtime_status.hpp>
#include <motors/srv/clear_errors.hpp>
#include <motors/srv/control_motor.hpp>
#include <motors/srv/identify_motor_id.hpp>
#include <motors/srv/read_motors.hpp>
#include <motors/srv/reset_motors.hpp>
#include <motors/srv/run_safety_check.hpp>
#include <motors/srv/scan_motors.hpp>
#include <motors/srv/set_master_id.hpp>
#include <motors/srv/set_motor_gains.hpp>
#include <motors/srv/set_operational_state.hpp>
#include <motors/srv/set_zeros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/eventfd.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace esd_link_bridge {
namespace {
using protocol::ActuatorCommand;
using SteadyClock = std::chrono::steady_clock;
constexpr auto kPortConfigRetry = std::chrono::seconds(2);
constexpr int kPortConfigRetryLimit = 2;

std::uint64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             SteadyClock::now().time_since_epoch()).count();
}

struct StateEnvelope {
  protocol::RobotState state{};
  protocol::DeviceStatus device{};
  std::uint64_t receive_ns{0};
  std::uint64_t parse_done_ns{0};
};

struct PendingTransaction {
  std::uint32_t transaction_id{0};
  protocol::Frame frame{};
  std::mutex mutex;
  std::condition_variable condition;
  bool done{false};
  protocol::TransactionResult result{};
};

struct QueuedCommand {
  std::uint8_t expected_mode{0};
  std::uint32_t source_state_sequence{0};
  std::uint64_t origin_ns{0};
  std::array<ActuatorCommand, protocol::kMaxPortCount> ports{};
  std::size_t count{0};
};

constexpr std::uint8_t kControlDisabled = 2;

ActuatorCommand *find_command(QueuedCommand &command, std::uint8_t port_id) {
  for (std::size_t i = 0; i < command.count; ++i) {
    if (command.ports[i].port_id == port_id) return &command.ports[i];
  }
  return nullptr;
}

constexpr std::uint8_t kModeDisabled = 0;
constexpr std::uint8_t kModeStandby = 1;
constexpr std::uint8_t kModePolicy = 2;
constexpr std::uint8_t kModeSweep = 3;
constexpr std::uint8_t kModeManual = 4;
constexpr std::uint8_t kModeFault = 5;
// Handshake/enable can still deliver one leftover DEVICE_STATUS or RobotState
// snapshot. Ignore it only for this window after the host opens the output
// stream; later dirty frames still latch immediately.
constexpr std::uint64_t kOutputArmingGraceNs = 400000000ULL;

}  // namespace

class EsdLinkBridgeNode : public rclcpp::Node {
 public:
  EsdLinkBridgeNode() : Node("esd_link_bridge_node"), decoder_(
      [this](const protocol::Frame &frame, std::uint64_t time_ns) { on_frame(frame, time_ns); }) {
    declare_parameters();
    load_parameters();
    create_ros_interfaces();
    if (::mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      RCLCPP_WARN(get_logger(), "mlockall failed: %s", std::strerror(errno));
    }
    wake_fd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (wake_fd_ < 0) throw std::runtime_error("eventfd failed: " + std::string(std::strerror(errno)));
    running_.store(true);
    io_thread_ = std::thread(&EsdLinkBridgeNode::io_loop, this);
    publish_thread_ = std::thread(&EsdLinkBridgeNode::publish_loop, this);
    RCLCPP_INFO(
        get_logger(),
        "ESD-Link bridge started read-only: device=%s match=%s, auto_enable=false",
        serial_device_.c_str(), serial_device_match_.c_str());
  }

  ~EsdLinkBridgeNode() override {
    request_disable_best_effort();
    running_.store(false);
    wake_io();
    publish_condition_.notify_all();
    if (io_thread_.joinable()) io_thread_.join();
    if (publish_thread_.joinable()) publish_thread_.join();
    close_serial();
    if (wake_fd_ >= 0) ::close(wake_fd_);
  }

 private:
  void declare_parameters() {
    declare_parameter<std::string>("serial_device", "auto");
    declare_parameter<std::string>("serial_device_match", "ESD-SLAVE");
    declare_parameter<std::string>("serial_by_id_directory", "/dev/serial/by-id");
    declare_parameter<std::string>("robot_profile_path", "");
    declare_parameter<int>("expected_control_rate_hz", 500);
    declare_parameter<std::string>("identity_record_path", "");
    declare_parameter<int>("realtime_priority", 70); declare_parameter<int>("cpu_affinity", -1);
    declare_parameter<int>("enable_confirm_timeout_ms", 1000);
    declare_parameter<int>("disable_confirm_timeout_ms", 6500);
    declare_parameter<bool>("auto_enable", false);
    declare_parameter<bool>("enforce_config_fingerprint", true);
  }
  void load_parameters() {
    serial_device_=get_parameter("serial_device").as_string(); if(serial_device_.empty())throw std::runtime_error("serial_device is required");
    serial_device_match_=get_parameter("serial_device_match").as_string();
    serial_by_id_directory_=get_parameter("serial_by_id_directory").as_string();
    if(serial_device_=="auto"&&serial_device_match_.empty())throw std::runtime_error("serial_device_match is required when serial_device=auto");
    if(serial_by_id_directory_.empty())throw std::runtime_error("serial_by_id_directory is required");
    robot_profile_path_=get_parameter("robot_profile_path").as_string();
    if(robot_profile_path_.empty())throw std::runtime_error("robot_profile_path is required");
    std::string config_error;
    if(!load_robot_profile(robot_profile_path_,profile_,config_error))throw std::runtime_error(config_error);
    motor_table_=profile_.motor_table;
    layout_id_=profile_.layout_id;
    schema_id_=profile_.schema_id;
    expected_mask_=motor_table_.mask;
    const auto expected_rate_parameter=get_parameter("expected_control_rate_hz").as_int();if(expected_rate_parameter<=0||expected_rate_parameter>2000)throw std::runtime_error("expected_control_rate_hz must be in (0, 2000]");expected_rate_=static_cast<std::uint16_t>(expected_rate_parameter);
    identity_record_path_=get_parameter("identity_record_path").as_string();
    rt_priority_=get_parameter("realtime_priority").as_int();cpu_affinity_=get_parameter("cpu_affinity").as_int();enable_confirm_timeout_ms_=get_parameter("enable_confirm_timeout_ms").as_int();disable_confirm_timeout_ms_=get_parameter("disable_confirm_timeout_ms").as_int();standby_ramp_s_=profile_.standby_ramp_seconds;
    for(std::size_t i=0;i<6;++i){const auto*actuator=find_actuator(profile_,static_cast<std::uint8_t>(i+1));if(!actuator)continue;defaults_[i]=actuator->default_position_rad;sign_[i]=actuator->direction;velocity_only_[i]=actuator->control==ActuatorControl::VELOCITY;kp_[i]=actuator->kp;kd_[i]=actuator->kd;default_kp_[i]=actuator->kp;default_kd_[i]=actuator->kd;if(actuator->has_position_limits){limits_[2*i]=actuator->position_min_rad-actuator->default_position_rad;limits_[2*i+1]=actuator->position_max_rad-actuator->default_position_rad;}else{limits_[2*i]=-6.28F;limits_[2*i+1]=6.28F;}if(i==2)wheel_velocity_limits_[0]=actuator->command_velocity_max_rad_s;if(i==5)wheel_velocity_limits_[1]=actuator->command_velocity_max_rad_s;}
    for(std::size_t i=0;i<2;++i){const auto*actuator=find_actuator(profile_,static_cast<std::uint8_t>(7+i));if(!actuator)continue;wing_sign_[i]=actuator->direction;wing_limits_[2*i]=actuator->position_min_rad;wing_limits_[2*i+1]=actuator->position_max_rad;}
    state_mapper_=std::make_unique<RobotStateMapper>(profile_);
    command_arbiter_=std::make_unique<CommandArbiter>(profile_);
    if(get_parameter("auto_enable").as_bool())throw std::runtime_error("auto_enable is forbidden for frozen ESD-Link hardware");
    enforce_config_fingerprint_=get_parameter("enforce_config_fingerprint").as_bool();
    if(!enforce_config_fingerprint_){
      std::fprintf(stderr,"esd_link_io: config fingerprint enforcement disabled; session uses the live lower fingerprint\n");
      std::fflush(stderr);
    }
    RCLCPP_INFO(get_logger(),"Robot Profile loaded: id=%s sha256=%s ros_hash=%016llx ports=%zu control_allowed=%s calibrated=%s",profile_.profile_id.c_str(),profile_.canonical_sha256.c_str(),static_cast<unsigned long long>(profile_.profile_hash),profile_.actuators.size(),profile_.control_allowed?"true":"false",profile_.calibrated?"true":"false");
  }

  void create_ros_interfaces() {
    auto sensor_qos=rclcpp::SensorDataQoS();auto latched=rclcpp::QoS(1).reliable().transient_local();
    lower_pub_=create_publisher<esd_link_msgs::msg::LowerState>("/lower/state",sensor_qos);
    robot_pub_=create_publisher<esd_link_msgs::msg::RobotState>("/robot/state",sensor_qos);
    final_command_pub_=create_publisher<esd_link_msgs::msg::ActuatorCommand>("/control/final_command",rclcpp::QoS(10).reliable());
    link_pub_=create_publisher<esd_link_msgs::msg::LinkStatus>("/lower/link_status",latched);
    policy_actuator_sub_=create_subscription<esd_link_msgs::msg::ActuatorCommand>("/control/policy_command",rclcpp::QoS(1).reliable(),[this](esd_link_msgs::msg::ActuatorCommand::SharedPtr msg){
      // Inference is deliberately allowed to warm up while STANDBY is active.
      // Those frames are not command rejections and must never reach the lower controller.
      if(control_mode_.load()==kModeStandby)return;
      actuator_contribution(*msg);
    });
    auxiliary_actuator_sub_=create_subscription<esd_link_msgs::msg::ActuatorCommand>("/control/auxiliary_command",rclcpp::QoS(1).reliable(),[this](esd_link_msgs::msg::ActuatorCommand::SharedPtr msg){cache_auxiliary_contribution(*msg);});
    maintenance_actuator_sub_=create_subscription<esd_link_msgs::msg::ActuatorCommand>("/control/maintenance_command",rclcpp::QoS(1).reliable(),[this](esd_link_msgs::msg::ActuatorCommand::SharedPtr msg){actuator_contribution(*msg);});
    joy_sub_=create_subscription<sensor_msgs::msg::Joy>("/joy",sensor_qos,[this](sensor_msgs::msg::Joy::SharedPtr msg){joy_command(*msg);});
    control_mode_service_=create_service<esd_link_msgs::srv::SetControlMode>("/lower/set_control_mode",[this](std::shared_ptr<esd_link_msgs::srv::SetControlMode::Request> req,std::shared_ptr<esd_link_msgs::srv::SetControlMode::Response> res){set_control_mode(req->target_mode,res->success,res->actual_mode,res->message);});
    zero_native_service_=create_service<esd_link_msgs::srv::SetLowerZeros>("/lower/set_zeros",[this](std::shared_ptr<esd_link_msgs::srv::SetLowerZeros::Request> req,std::shared_ptr<esd_link_msgs::srv::SetLowerZeros::Response> res){native_set_zeros(*req,*res);});
    operational_service_=create_service<motors::srv::SetOperationalState>("/motors/set_operational_state",[this](std::shared_ptr<motors::srv::SetOperationalState::Request> req,std::shared_ptr<motors::srv::SetOperationalState::Response> res){if(req->target_state>motors::srv::SetOperationalState::Request::ACTIVE){res->success=false;res->actual_state=compat_motor_state(control_mode_.load());res->message="invalid operational state";return;}const std::uint8_t target=req->target_state==motors::srv::SetOperationalState::Request::DISARMED?kModeDisabled:(req->target_state==motors::srv::SetOperationalState::Request::STANDBY?kModeStandby:kModePolicy);std::uint8_t actual_mode=control_mode_.load();set_control_mode(target,res->success,actual_mode,res->message);res->actual_state=compat_motor_state(actual_mode);});
    soft_disarm_service_=create_service<std_srvs::srv::Trigger>("/motors/soft_disarm",[this](std::shared_ptr<std_srvs::srv::Trigger::Request>,std::shared_ptr<std_srvs::srv::Trigger::Response> res){std::uint8_t actual=0;set_control_mode(kModeDisabled,res->success,actual,res->message);});
    zero_compat_service_=create_service<motors::srv::SetZeros>("/set_zeros",[this](std::shared_ptr<motors::srv::SetZeros::Request>,std::shared_ptr<motors::srv::SetZeros::Response> res){esd_link_msgs::srv::SetLowerZeros::Request req;req.persist=true;req.port_ids={1,2,3,4,5,6};req.assigned_position_rad.assign(6,0.0F);esd_link_msgs::srv::SetLowerZeros::Response native;native_set_zeros(req,native);res->success=native.success;res->message=native.message;});
    gains_service_=create_service<motors::srv::SetMotorGains>("/set_motor_gains",[this](std::shared_ptr<motors::srv::SetMotorGains::Request> req,std::shared_ptr<motors::srv::SetMotorGains::Response> res){set_gains(*req,*res);});
    wing_rc_service_=create_service<std_srvs::srv::SetBool>("/wing/enable_rc",[this](std::shared_ptr<std_srvs::srv::SetBool::Request> req,std::shared_ptr<std_srvs::srv::SetBool::Response> res){wing_rc_enabled_.store(req->data);res->success=true;res->message=req->data?"wing RC enabled":"wing RC disabled; holding current target";});
    read_service_=create_service<motors::srv::ReadMotors>("/read_motors",[this](std::shared_ptr<motors::srv::ReadMotors::Request>,std::shared_ptr<motors::srv::ReadMotors::Response> res){read_motors(*res);});
    control_service_=create_service<motors::srv::ControlMotor>("/control_motor",[this](std::shared_ptr<motors::srv::ControlMotor::Request> req,std::shared_ptr<motors::srv::ControlMotor::Response> res){manual_control(*req,*res);});
    unsupported_reset_=unsupported_service<motors::srv::ResetMotors>("/reset_motors"); unsupported_clear_=unsupported_service<motors::srv::ClearErrors>("/clear_errors"); unsupported_identify_=unsupported_service<motors::srv::IdentifyMotorID>("/identify_motor_id"); unsupported_master_id_=unsupported_service<motors::srv::SetMasterID>("/set_master_id"); unsupported_scan_=create_service<motors::srv::ScanMotors>("/scan_motors",[](std::shared_ptr<motors::srv::ScanMotors::Request>,std::shared_ptr<motors::srv::ScanMotors::Response> res){res->success=false;res->message="unsupported by frozen ESD-Link firmware";res->detected_motor_ids.clear();}); unsupported_safety_=unsupported_service<motors::srv::RunSafetyCheck>("/run_safety_check");
    diagnostics_timer_=create_wall_timer(std::chrono::seconds(1),[this]{publish_link_status();});
  }

  template<typename ServiceT>
  typename rclcpp::Service<ServiceT>::SharedPtr unsupported_service(const std::string&name){return create_service<ServiceT>(name,[](std::shared_ptr<typename ServiceT::Request>,std::shared_ptr<typename ServiceT::Response> res){res->success=false;res->message="unsupported by frozen ESD-Link firmware";});}

  static std::uint8_t compat_motor_state(std::uint8_t mode){return mode==kModeDisabled?motors::msg::MotorRuntimeStatus::DISARMED:(mode==kModeStandby?motors::msg::MotorRuntimeStatus::STANDBY:(mode==kModePolicy||mode==kModeSweep||mode==kModeManual?motors::msg::MotorRuntimeStatus::ACTIVE:motors::msg::MotorRuntimeStatus::FAULT));}

  bool persist_identity_record(std::uint32_t config_fingerprint,std::string&error){if(identity_record_path_.empty())return true;try{const std::filesystem::path path(identity_record_path_);if(path.has_parent_path())std::filesystem::create_directories(path.parent_path());const auto temporary=path.string()+".tmp";{std::ofstream stream(temporary,std::ios::trunc);if(!stream)throw std::runtime_error("cannot open temporary identity record");stream<<"schema_id="<<schema_id_<<"\nlayout_id="<<layout_id_<<"\nactive_port_mask=0x"<<std::hex<<std::setw(8)<<std::setfill('0')<<expected_mask_<<"\nconfig_fingerprint=0x"<<std::setw(8)<<config_fingerprint<<"\n";stream.flush();if(!stream)throw std::runtime_error("cannot flush temporary identity record");}std::filesystem::rename(temporary,path);return true;}catch(const std::exception&exception){error=exception.what();return false;}}

  void configure_realtime() {
    pthread_setname_np(pthread_self(),"esd_link_io");
    if(cpu_affinity_>=0){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu_affinity_,&set);const int result=pthread_setaffinity_np(pthread_self(),sizeof(set),&set);if(result!=0)std::fprintf(stderr,"esd_link_io: cannot set CPU affinity: %s\n",std::strerror(result));}
    if(rt_priority_>0){sched_param p{};p.sched_priority=rt_priority_;const int result=pthread_setschedparam(pthread_self(),SCHED_FIFO,&p);if(result!=0)std::fprintf(stderr,"esd_link_io: cannot set SCHED_FIFO %d: %s\n",rt_priority_,std::strerror(result));}
  }
  bool open_serial() {
    const auto resolution=resolve_serial_device(serial_device_,serial_device_match_,serial_by_id_directory_);
    if(!resolution){if(resolution.error!=last_serial_error_){std::fprintf(stderr,"esd_link_io: %s\n",resolution.error.c_str());last_serial_error_=resolution.error;}return false;}
    active_serial_device_=resolution.path;
    fd_=::open(active_serial_device_.c_str(),O_RDWR|O_NOCTTY|O_NONBLOCK|O_CLOEXEC);if(fd_<0){const auto message="cannot open "+active_serial_device_+": "+std::strerror(errno);if(message!=last_serial_error_){std::fprintf(stderr,"esd_link_io: %s\n",message.c_str());last_serial_error_=message;}return false;}
    if(::flock(fd_,LOCK_EX|LOCK_NB)!=0){::close(fd_);fd_=-1;return false;}
    termios tty{};if(tcgetattr(fd_,&tty)!=0){close_serial();return false;}cfmakeraw(&tty);cfsetispeed(&tty,B115200);cfsetospeed(&tty,B115200);tty.c_cflag|=CLOCAL|CREAD;tty.c_cflag&=~CRTSCTS;tty.c_cc[VMIN]=0;tty.c_cc[VTIME]=0;if(tcsetattr(fd_,TCSANOW,&tty)!=0){close_serial();return false;}
    int modem=0;if(::ioctl(fd_,TIOCMGET,&modem)==0){modem|=TIOCM_DTR;modem&=~TIOCM_RTS;(void)::ioctl(fd_,TIOCMSET,&modem);}
    tcflush(fd_,TCIOFLUSH);decoder_.reset();connected_.store(true);session_ready_.store(false);config_in_flight_.store(false);verified_fingerprint_.store(0);session_id_.store(0);config_retries_=0;open_attempts_=0;config_busy_seen_.store(false);last_serial_error_.clear();return true;
  }
  void close_serial(){if(fd_>=0){::flock(fd_,LOCK_UN);::close(fd_);fd_=-1;}connected_.store(false);session_ready_.store(false);config_in_flight_.store(false);verified_fingerprint_.store(0);session_id_.store(0);config_retries_=0;open_attempts_=0;config_busy_seen_.store(false);have_policy_.store(false);have_policy_source_.store(false);have_maintenance_source_.store(false);standby_active_.store(false);{std::lock_guard<std::mutex>lock(cache_mutex_);have_state_=false;}{std::lock_guard<std::mutex>lock(robot_state_mutex_);have_robot_state_=false;}{std::lock_guard<std::mutex>lock(auxiliary_mutex_);auxiliary_initialized_mask_=0;latest_auxiliary_contribution_.reset();}lower_control_state_.store(0);control_state_condition_.notify_all();standby_condition_.notify_all();if(control_mode_.exchange(kModeDisabled)!=kModeDisabled)std::fprintf(stderr,"esd_link_io: disconnected; control mode forced DISABLED\n");}
  bool write_wire(const protocol::WireBuffer&w){std::size_t off=0;while(off<w.size&&running_.load()){const auto n=::write(fd_,w.bytes.data()+off,w.size-off);if(n>0){off+=static_cast<std::size_t>(n);continue;}if(n<0&&(errno==EAGAIN||errno==EINTR)){pollfd p{fd_,POLLOUT,0};if(::poll(&p,1,10)>=0)continue;}return false;}return off==w.size;}
  bool send_frame(protocol::Frame frame){frame.sequence=++tx_sequence_;protocol::WireBuffer wire;if(!protocol::encode_frame(frame,wire))return false;return write_wire(wire);}
  void send_open(){
    std::random_device rd;host_nonce_=static_cast<std::uint32_t>(rd())^static_cast<std::uint32_t>(monotonic_ns());if(!host_nonce_)host_nonce_=1;
    protocol::Frame f;
    if(!protocol::build_open_session(host_nonce_,layout_id_,schema_id_,0,f))return;
    send_frame(f);
    ++open_attempts_;
    if(open_attempts_==1||open_attempts_%5==0){
      std::fprintf(stderr,
        "esd_link_io: OPEN_SESSION attempt=%u frames=%u cobs=%u crc=%u\n",
        open_attempts_,decoder_.frames(),decoder_.cobs_errors(),decoder_.crc_errors());
      std::fflush(stderr);
    }
  }
  void send_port_config(){
    if(config_in_flight_.load())return;
    protocol::Frame frame;
    const auto tx=++transaction_sequence_;
    config_transaction_id_=tx;
    if(!protocol::build_port_config_update(session_id_.load(),tx,motor_table_.entries.data(),motor_table_.count,frame)){
      std::fprintf(stderr,"esd_link_io: cannot build PORT_CONFIG_UPDATE\n");
      return;
    }
    frame.sequence=++tx_sequence_;
    if(!protocol::encode_frame(frame,config_wire_)){
      std::fprintf(stderr,"esd_link_io: cannot encode PORT_CONFIG_UPDATE\n");
      return;
    }
    config_in_flight_.store(true);
    config_busy_seen_.store(false);
    config_retries_=0;
    config_retry_at_=SteadyClock::now()+kPortConfigRetry;
    write_wire(config_wire_);
    std::fprintf(stderr,"esd_link_io: sending PORT_CONFIG_UPDATE ports=%zu mask=0x%08x fingerprint=0x%08x expected=0x%08x\n",
                 motor_table_.count,expected_mask_,fingerprint_.load(),profile_.lower_config_fingerprint);
    std::fflush(stderr);
  }
  void retry_port_config_if_needed(){
    if(!config_in_flight_.load()||!config_busy_seen_.load()||fd_<0||SteadyClock::now()<config_retry_at_)return;
    if(config_retries_>=kPortConfigRetryLimit)return;
    ++config_retries_;
    config_retry_at_=SteadyClock::now()+kPortConfigRetry;
    write_wire(config_wire_);
    std::fprintf(stderr,"esd_link_io: retrying PORT_CONFIG_UPDATE %d/%d\n",config_retries_,kPortConfigRetryLimit);
    std::fflush(stderr);
  }

  void io_loop(){configure_realtime();auto next_open=SteadyClock::now();auto next_handshake=SteadyClock::now();std::array<std::uint8_t,2048> read_buffer{};while(running_.load()){
      if(fd_<0){if(SteadyClock::now()>=next_open){if(open_serial()){reconnects_.fetch_add(1);send_open();next_handshake=SteadyClock::now()+std::chrono::seconds(1);std::fprintf(stderr,"esd_link_io: opened %s\n",active_serial_device_.c_str());}next_open=SteadyClock::now()+std::chrono::seconds(1);}std::this_thread::sleep_for(std::chrono::milliseconds(100));continue;}
      retry_port_config_if_needed();
      if(reopen_session_.exchange(false)||(!session_ready_.load()&&!config_in_flight_.load()&&SteadyClock::now()>=next_handshake)){send_open();next_handshake=SteadyClock::now()+std::chrono::seconds(1);}
      pollfd fds[2]={{fd_,POLLIN|POLLERR|POLLHUP,0},{wake_fd_,POLLIN,0}};const int result=::poll(fds,2,100);
      if(result<0){if(errno==EINTR)continue;close_serial();continue;}
      if(fds[1].revents&POLLIN){std::uint64_t count;while(::read(wake_fd_,&count,sizeof(count))>0){}drain_outbound();}
      if(fds[0].revents&(POLLERR|POLLHUP|POLLNVAL)){close_serial();continue;}
      if(fds[0].revents&POLLIN){for(;;){const auto n=::read(fd_,read_buffer.data(),read_buffer.size());if(n>0){decoder_.feed(read_buffer.data(),static_cast<std::size_t>(n),monotonic_ns());continue;}if(n<0&&errno!=EAGAIN&&errno!=EINTR)close_serial();break;}}
    }}
  void drain_outbound(){std::deque<std::shared_ptr<PendingTransaction>> transactions;std::optional<QueuedCommand> command;{std::lock_guard<std::mutex> lock(outbound_mutex_);transactions.swap(transactions_);command.swap(queued_command_);}for(auto&t:transactions)if(fd_>=0&&session_id_.load())send_frame(t->frame);if(command&&!disable_in_progress_.load()&&fd_>=0&&session_ready_.load()&&command->expected_mode==control_mode_.load()){protocol::Frame f;if(protocol::build_actuator_command(session_id_.load(),layout_id_,command->source_state_sequence,fingerprint_.load(),command->ports.data(),command->count,f)){const auto seq=tx_sequence_.load()+1;if(send_frame(f)){const auto sent=monotonic_ns();std::lock_guard<std::mutex>lock(ack_mutex_);const auto&state_time=state_times_[command->source_state_sequence%state_times_.size()];const auto observation_ns=state_time.sequence==command->source_state_sequence?state_time.receive_ns:0;acks_[seq%acks_.size()]={seq,sent,observation_ns};if(observation_ns&&sent>=observation_ns)last_observation_to_command_ns_.store(sent-observation_ns);last_sent_command_.store(seq);}}}}

  bool host_command_stream_active() const {
    switch (control_mode_.load()) {
      case kModeStandby:
      case kModePolicy:
      case kModeSweep:
      case kModeManual:
        return true;
      default:
        return false;
    }
  }

  void begin_output_arming_grace() {
    output_arming_grace_until_ns_.store(monotonic_ns() + kOutputArmingGraceNs);
  }

  bool output_arming_grace_active() const {
    const auto until = output_arming_grace_until_ns_.load();
    return until != 0 && monotonic_ns() < until;
  }

  void observe_lower_control_state(std::uint8_t state) {
    lower_control_state_.store(state);
    device_control_state_.store(state);
    control_state_generation_.fetch_add(1);
    control_state_condition_.notify_all();
    if (output_arming_grace_active()) return;
    if (state == 6) {
      if (host_command_stream_active()) fault_from_io();
      return;
    }
    if (!disable_in_progress_.load() && (state == 2 || state == 5)) {
      const auto mode = control_mode_.load();
      if (mode != kModeDisabled && mode != kModeFault) {
        control_mode_.store(kModeDisabled);
        have_policy_.store(false);
        have_policy_source_.store(false);
        have_maintenance_source_.store(false);
        standby_active_.store(false);
        standby_condition_.notify_all();
      }
    }
  }

  void observe_device_status(const protocol::DeviceStatus& device) {
    const bool output_armed = host_command_stream_active();
    observe_lower_control_state(device.control_state);
    device_valid_commands_.store(device.valid_command_count);
    device_invalid_frames_.store(device.invalid_frame_count);
    device_rejected_commands_.store(device.rejected_command_count);
    device_command_age_ms_.store(device.command_age_ms);
    last_reject_code_.store(device.last_reject_code);
    const auto previous_faults = fault_flags_.exchange(device.fault_flags);
    if ((device.fault_flags & 2U) && !(previous_faults & 2U)) watchdog_events_.fetch_add(1);
    if ((device.fault_flags & 0x80U) && !(previous_faults & 0x80U)) control_fault_events_.fetch_add(1);
    const auto previous_offline = offline_mask_.exchange(device.offline_port_mask);
    if (device.offline_port_mask != previous_offline || device.fault_flags != previous_faults) {
      std::fprintf(stderr, "esd_link_io: DEVICE_STATUS offline_port_mask=0x%08x ports=[",
                   device.offline_port_mask);
      bool first = true;
      for (unsigned port = 1; port < 32; ++port) {
        if (device.offline_port_mask & (1u << port)) {
          std::fprintf(stderr, "%s%u", first ? "" : ",", port);
          first = false;
        }
      }
      std::fprintf(stderr,
                   "] fault_flags=0x%08x last_reject=%s(0x%04x) control_state=%u active=0x%08x\n",
                   device.fault_flags, protocol::status_name(device.last_reject_code),
                   device.last_reject_code, device.control_state, device.active_port_mask);
      std::fflush(stderr);
    }
    if (output_armed && (device.fault_flags || device.offline_port_mask || device.control_state == 6)) {
      if (output_arming_grace_active()) {
        std::fprintf(stderr,
                     "esd_link_io: ignoring transient DEVICE_STATUS during output arming grace "
                     "offline_port_mask=0x%08x fault_flags=0x%08x control_state=%u\n",
                     device.offline_port_mask, device.fault_flags, device.control_state);
        std::fflush(stderr);
      } else {
        fault_from_io();
      }
    }
  }

  void on_frame(const protocol::Frame&frame,std::uint64_t receive_ns){if(frame.kind==protocol::Kind::Response){if(protocol::response_status(frame)==protocol::kStatusBusy){if(config_in_flight_.load()){config_busy_seen_.store(true);config_retry_at_=SteadyClock::now()+kPortConfigRetry;std::fprintf(stderr,"esd_link_io: PORT_CONFIG_UPDATE BUSY; waiting for final response then retrying the same frame\n");std::fflush(stderr);}return;}protocol::SessionInfo session;if(protocol::parse_session_info(frame,session)){handle_session(session);return;}protocol::TransactionResult result;if(protocol::parse_transaction_result(frame,result)&&result.status!=0xFFFF){if(config_in_flight_.load()&&result.transaction_id==config_transaction_id_){handle_port_config_result(result);return;}complete_transaction(result);return;}return;}
    protocol::RobotState state;if(protocol::parse_robot_state(frame,state)){if(!session_ready_.load()||state.session_id!=session_id_.load()||state.schema_id!=schema_id_)return;observe_lower_control_state(state.control_state);const auto parsed=monotonic_ns();StateEnvelope envelope;envelope.state=state;envelope.receive_ns=receive_ns;envelope.parse_done_ns=parsed;update_state_stats(state,receive_ns);state_mailbox_.publish(envelope);publish_condition_.notify_one();process_ack(state.last_applied_command_seq,receive_ns);return;}
    protocol::DeviceStatus device;if(protocol::parse_device_status(frame,device)&&device.session_id==session_id_.load()){observe_device_status(device);return;}}
  void handle_session(const protocol::SessionInfo&s){
    if(s.status||s.host_nonce!=host_nonce_||s.business_version!=1||s.layout_id!=layout_id_||s.schema_id!=schema_id_||s.control_rate_hz!=expected_rate_){
      std::fprintf(stderr,"esd_link_io: session mismatch status=%s layout=%u schema=%u mask=0x%08x rate=%u\n",protocol::status_name(s.status),s.layout_id,s.schema_id,s.active_port_mask,s.control_rate_hz);
      session_ready_.store(false);
      return;
    }
    session_id_.store(s.session_id);
    fingerprint_.store(s.config_fingerprint);
    lower_boot_id_.store(s.lower_boot_id);
    observe_lower_control_state(s.control_state);
    fault_flags_.store(0);
    offline_mask_.store(0);
    last_reject_code_.store(0);
    last_state_sequence_.store(0);
    last_state_ns_.store(0);
    max_gap_ns_.store(0);
    last_status_state_count_.store(state_count_.load());
    have_policy_source_.store(false);
    have_maintenance_source_.store(false);
    {std::lock_guard<std::mutex>lock(cache_mutex_);have_state_=false;}
    {std::lock_guard<std::mutex>lock(ack_mutex_);acks_.fill({});state_times_.fill({});}
    if(s.active_port_mask!=expected_mask_){
      session_ready_.store(false);
      if(s.control_state==kControlDisabled) send_port_config();
      else std::fprintf(stderr,"esd_link_io: waiting DISABLED before PORT_CONFIG_UPDATE mask=0x%08x expected=0x%08x\n",s.active_port_mask,expected_mask_);
      return;
    }
    const auto verified_fingerprint=verified_fingerprint_.load();
    const auto expected_fingerprint=verified_fingerprint!=0U?
      verified_fingerprint:profile_.lower_config_fingerprint;
    if(enforce_config_fingerprint_&&expected_fingerprint!=0U&&s.config_fingerprint!=expected_fingerprint){
      session_ready_.store(false);
      if(s.control_state==kControlDisabled){
        std::fprintf(stderr,
          "esd_link_io: fingerprint mismatch actual=0x%08x expected=0x%08x; verifying full port table\n",
          s.config_fingerprint,expected_fingerprint);
        send_port_config();
      }else{
        std::fprintf(stderr,
          "esd_link_io: fingerprint mismatch actual=0x%08x expected=0x%08x; waiting DISABLED\n",
          s.config_fingerprint,expected_fingerprint);
      }
      std::fflush(stderr);
      return;
    }
    if(!enforce_config_fingerprint_&&expected_fingerprint!=0U&&s.config_fingerprint!=expected_fingerprint){
      std::fprintf(stderr,
        "esd_link_io: accepting live fingerprint 0x%08x (profile expected 0x%08x; enforcement off)\n",
        s.config_fingerprint,expected_fingerprint);
      std::fflush(stderr);
    }
    session_ready_.store(true);
    open_attempts_=0;
    std::fprintf(stderr,"esd_link_io: session ready session=%u boot=%u fingerprint=0x%08x mask=0x%08x\n",s.session_id,s.lower_boot_id,s.config_fingerprint,s.active_port_mask);
    std::fflush(stderr);
  }
  void handle_port_config_result(const protocol::TransactionResult&r){
    config_in_flight_.store(false);
    config_busy_seen_.store(false);
    config_retries_=0;
    if(r.status){
      std::fprintf(stderr,"esd_link_io: PORT_CONFIG_UPDATE rejected: %s\n",protocol::status_name(r.status));
      return;
    }
    const bool masks_verified=
      r.requested_port_mask==expected_mask_&&
      r.affected_port_mask==expected_mask_&&
      r.verified_port_mask==expected_mask_&&
      r.rollback_port_mask==0U;
    if(!masks_verified||(r.detail_code!=0&&r.detail_code!=1)){
      session_ready_.store(false);
      std::fprintf(stderr,
        "esd_link_io: PORT_CONFIG_UPDATE verification failed detail=%u requested=0x%08x affected=0x%08x verified=0x%08x rollback=0x%08x\n",
        r.detail_code,r.requested_port_mask,r.affected_port_mask,
        r.verified_port_mask,r.rollback_port_mask);
      std::fflush(stderr);
      send_open();
      return;
    }
    verified_fingerprint_.store(r.config_fingerprint);
    fingerprint_.store(r.config_fingerprint);
    if(r.detail_code==0){
      session_id_.store(0);
      session_ready_.store(false);
      std::fprintf(stderr,"esd_link_io: port table changed; reopening session fingerprint=0x%08x\n",r.config_fingerprint);
      std::fflush(stderr);
      send_open();
      return;
    }
    if(r.detail_code==1){
      session_ready_.store(true);
      std::fprintf(stderr,"esd_link_io: port table unchanged; session kept fingerprint=0x%08x\n",r.config_fingerprint);
      std::fflush(stderr);
      return;
    }
    session_ready_.store(false);
    send_open();
  }
  void complete_transaction(const protocol::TransactionResult&r){std::shared_ptr<PendingTransaction> match;{std::lock_guard<std::mutex>lock(pending_mutex_);auto it=std::find_if(pending_.begin(),pending_.end(),[&](const auto&p){return p->transaction_id==r.transaction_id;});if(it!=pending_.end()){match=*it;pending_.erase(it);}}if(match){std::lock_guard<std::mutex>lock(match->mutex);match->result=r;match->done=true;match->condition.notify_all();}}
  void update_state_stats(const protocol::RobotState&s,std::uint64_t now){state_count_.fetch_add(1);const auto last=last_state_ns_.exchange(now);if(last){const auto gap=now-last;auto current=max_gap_ns_.load();while(gap>current&&!max_gap_ns_.compare_exchange_weak(current,gap)){};}const auto previous=last_state_sequence_.exchange(s.state_sample_seq);if(previous){const auto delta=s.state_sample_seq-previous;if(delta>1&&delta<0x80000000U)sequence_gaps_.fetch_add(delta-1);}std::lock_guard<std::mutex>lock(ack_mutex_);state_times_[s.state_sample_seq%state_times_.size()]={s.state_sample_seq,now};}
  struct AckEntry{std::uint32_t sequence{0};std::uint64_t sent_ns{0};std::uint64_t observation_ns{0};};
  struct StateTime{std::uint32_t sequence{0};std::uint64_t receive_ns{0};};
  void process_ack(std::uint32_t seq,std::uint64_t now){if(!seq)return;last_applied_sequence_.store(seq);std::lock_guard<std::mutex>lock(ack_mutex_);auto&e=acks_[seq%acks_.size()];if(e.sequence==seq&&e.sent_ns){last_apply_latency_ns_.store(now-e.sent_ns);e={};}}
  bool accept_new_source_sequence(
      std::uint32_t source, std::atomic<std::uint32_t>& last,
      std::atomic<bool>& have_last) {
    {
      std::lock_guard<std::mutex> lock(ack_mutex_);
      const auto& seen = state_times_[source % state_times_.size()];
      if (seen.sequence != source || seen.receive_ns == 0) {
        rejected_commands_.fetch_add(1);
        return false;
      }
    }
    if (have_last.load()) {
      const auto delta = static_cast<std::int32_t>(source - last.load());
      if (delta <= 0) {
        rejected_commands_.fetch_add(1);
        return false;
      }
    }
    last.store(source);
    have_last.store(true);
    return true;
  }

  void publish_loop(){pthread_setname_np(pthread_self(),"esd_link_pub");while(running_.load()){std::unique_lock<std::mutex>wait(publish_mutex_);publish_condition_.wait_for(wait,std::chrono::milliseconds(100));wait.unlock();StateEnvelope env;while(state_mailbox_.consume(env))publish_state(env);}}
  void publish_state(const StateEnvelope&e){{std::lock_guard<std::mutex>lock(cache_mutex_);latest_state_=e;have_state_=true;}
    esd_link_msgs::msg::LowerState msg;msg.header.stamp=now();msg.header.frame_id="lower_imu";msg.host_monotonic_ns=e.receive_ns;msg.parse_done_monotonic_ns=e.parse_done_ns;msg.session_id=e.state.session_id;msg.schema_id=e.state.schema_id;msg.device_sample_time_us=e.state.sample_time_us;msg.state_sample_seq=e.state.state_sample_seq;msg.last_applied_command_seq=e.state.last_applied_command_seq;msg.command_status_flags=e.state.command_status_flags;msg.control_state=e.state.control_state;msg.imu_valid_mask=e.state.imu_valid_mask;msg.fault_flags=fault_flags_.load();msg.active_port_mask=expected_mask_;msg.offline_port_mask=offline_mask_.load();
    msg.imu.header=msg.header;msg.imu.orientation.x=e.state.quaternion[0];msg.imu.orientation.y=e.state.quaternion[1];msg.imu.orientation.z=e.state.quaternion[2];msg.imu.orientation.w=e.state.quaternion[3];msg.imu.angular_velocity.x=e.state.gyro[0];msg.imu.angular_velocity.y=e.state.gyro[1];msg.imu.angular_velocity.z=e.state.gyro[2];msg.imu.linear_acceleration.x=e.state.accel[0];msg.imu.linear_acceleration.y=e.state.accel[1];msg.imu.linear_acceleration.z=e.state.accel[2];
    const auto n=e.state.port_count;msg.port_id.resize(n);msg.valid_mask.resize(n);msg.position_rad.resize(n);msg.velocity_rad_s.resize(n);msg.effort_nm.resize(n);
    for(std::size_t i=0;i<n;++i){msg.port_id[i]=e.state.ports[i].id;msg.valid_mask[i]=e.state.ports[i].valid_mask;msg.position_rad[i]=e.state.ports[i].position;msg.velocity_rad_s[i]=e.state.ports[i].velocity;msg.effort_nm[i]=e.state.ports[i].effort;}
    lower_pub_->publish(msg);
    esd_link_msgs::msg::RobotState robot;std::string mapping_error;if(state_mapper_->map(msg,robot,mapping_error)){advance_auxiliary_targets(robot);{std::lock_guard<std::mutex>lock(robot_state_mutex_);latest_robot_state_=robot;have_robot_state_=true;}robot_pub_->publish(robot);}else{RCLCPP_ERROR_THROTTLE(get_logger(),*get_clock(),1000,"RobotState mapping rejected: %s",mapping_error.c_str());}
    advance_standby(e);}
  const protocol::PortState*find_port(const protocol::RobotState&s,std::uint8_t id)const{for(std::size_t i=0;i<s.port_count;++i)if(s.ports[i].id==id)return&s.ports[i];return nullptr;}

  bool state_is_ready(StateEnvelope&state,std::string&reason){std::lock_guard<std::mutex>lock(cache_mutex_);if(!have_state_){reason="no lower state";return false;}state=latest_state_;if((state.state.imu_valid_mask&0x07U)!=0x07U){reason="IMU invalid";return false;}for(const auto value:state.state.gyro)if(!std::isfinite(value)){reason="IMU gyro non-finite";return false;}for(const auto value:state.state.quaternion)if(!std::isfinite(value)){reason="IMU quaternion non-finite";return false;}for(const auto value:state.state.accel)if(!std::isfinite(value)){reason="IMU acceleration non-finite";return false;}for(std::size_t i=0;i<motor_table_.count;++i){const auto id=motor_table_.entries[i].port_id;const auto*p=find_port(state.state,id);if(!p||(p->valid_mask&0x03U)!=0x03U||!std::isfinite(p->position)||!std::isfinite(p->velocity)||!std::isfinite(p->effort)){reason="port "+std::to_string(id)+" invalid";return false;}}if(lower_control_state_.load()>=5||fault_flags_.load()||offline_mask_.load()){reason="lower controller fault/offline";return false;}return true;}
  void capture_active_entry_positions(const StateEnvelope&state){active_entry_positions_.fill(0);for(std::size_t i=0;i<motor_table_.count;++i){const auto id=motor_table_.entries[i].port_id;const auto*p=find_port(state.state,id);if(!p)continue;if(id>=1&&id<=6)active_entry_positions_[id]=coordinates::policy_position_from_lower(p->position,sign_[id-1],defaults_[id-1]);else if(id==7||id==8)active_entry_positions_[id]=coordinates::policy_direction_from_lower(p->position,wing_sign_[id-7]);else active_entry_positions_[id]=p->position;}}
  bool maintenance_position_target_is_safe(std::uint8_t id,float target,float low,float high)const{if(target>=low&&target<=high)return true;const auto entry=active_entry_positions_[id];constexpr float tolerance=0.01F;if(entry>high)return target>=high&&target<=entry+tolerance;if(entry<low)return target<=low&&target>=entry-tolerance;return false;}
  std::pair<std::array<float,6>,std::array<float,6>> gains(){std::lock_guard<std::mutex>lock(gain_mutex_);return {kp_,kd_};}
  QueuedCommand hold_current(const StateEnvelope&s){QueuedCommand out{};for(const auto&actuator:profile_.actuators){const auto*p=find_port(s.state,actuator.port_id);auto&cmd=out.ports[out.count++];cmd.port_id=actuator.port_id;cmd.position=p?p->position:0;cmd.velocity=0;cmd.effort=0;cmd.kp=actuator.kp;cmd.kd=actuator.kd;}return out;}
  bool queue_command(std::uint8_t mode,std::uint32_t source,QueuedCommand command,std::uint64_t origin){for(std::size_t i=0;i<command.count;++i){const auto&p=command.ports[i];if(!std::isfinite(p.position)||!std::isfinite(p.velocity)||!std::isfinite(p.kp)||!std::isfinite(p.kd)||!std::isfinite(p.effort)||p.kp<0||p.kd<0){rejected_commands_.fetch_add(1);return false;}}command.expected_mode=mode;command.source_state_sequence=source;command.origin_ns=origin;{std::lock_guard<std::mutex>lock(outbound_mutex_);queued_command_=command;}wake_io();return true;}
  void wake_io(){if(wake_fd_>=0){std::uint64_t one=1;(void)::write(wake_fd_,&one,sizeof(one));}}
  std::shared_ptr<PendingTransaction> submit_transaction(protocol::Frame frame,std::uint32_t transaction_id,std::chrono::milliseconds timeout){auto pending=std::make_shared<PendingTransaction>();pending->transaction_id=transaction_id;frame.sequence=0;pending->frame=frame;{std::lock_guard<std::mutex>lock(pending_mutex_);pending_.push_back(pending);}{std::lock_guard<std::mutex>lock(outbound_mutex_);transactions_.push_back(pending);}wake_io();std::unique_lock<std::mutex>lock(pending->mutex);if(!pending->condition.wait_for(lock,timeout,[&]{return pending->done;})){std::lock_guard<std::mutex>p_lock(pending_mutex_);pending_.erase(std::remove(pending_.begin(),pending_.end(),pending),pending_.end());}return pending;}
  bool transact_enable(bool enable,std::string&message){if(!session_ready_.load()){message="ESD-Link session unavailable";return false;}std::array<std::uint8_t,protocol::kMaxPortCount> ids{};const auto count=protocol::fill_port_ids(expected_mask_,ids.data(),ids.size());protocol::Frame frame;const auto tx=++transaction_sequence_;if(!protocol::build_set_enable(session_id_.load(),tx,enable,ids.data(),count,frame)){message="cannot build SET_ENABLE";return false;}auto pending=submit_transaction(frame,tx,std::chrono::seconds(5));if(!pending->done){message="SET_ENABLE timeout";return false;}if(pending->result.status){message=std::string("SET_ENABLE rejected: ")+protocol::status_name(pending->result.status);return false;}const auto confirmation_generation=control_state_generation_.load();std::unique_lock<std::mutex>lock(control_state_mutex_);const auto confirm_timeout=std::chrono::milliseconds(enable?enable_confirm_timeout_ms_:disable_confirm_timeout_ms_);const bool confirmed=control_state_condition_.wait_for(lock,confirm_timeout,[this,enable,confirmation_generation]{const auto state=lower_control_state_.load();return control_state_generation_.load()!=confirmation_generation&&(enable?(state==3||state==4):state==2);});if(!confirmed){message=enable?"SET_ENABLE acknowledged but lower did not enter ENABLED_WAIT_COMMAND/ACTIVE":"SET_ENABLE acknowledged but lower did not finish SAFE_DAMPING and confirm DISABLED";return false;}message=enable?"configured ports enabled":"configured ports disabled";return true;}
  void queue_disable_no_wait(){if(!session_ready_.load()||fd_<0)return;protocol::Frame f;const auto transaction=++transaction_sequence_;if(protocol::build_set_enable(session_id_.load(),transaction,false,nullptr,0,f)){auto pending=std::make_shared<PendingTransaction>();pending->transaction_id=transaction;pending->frame=f;{std::lock_guard<std::mutex>lock(outbound_mutex_);transactions_.push_back(pending);}wake_io();}}
  void fault_from_io(){if(control_mode_.exchange(kModeFault)!=kModeFault){standby_active_.store(false);standby_condition_.notify_all();queue_disable_no_wait();}}
  void request_disable_best_effort(){queue_disable_no_wait();std::this_thread::sleep_for(std::chrono::milliseconds(50));}

  void set_control_mode(std::uint8_t target,bool&success,std::uint8_t&actual,std::string&message){std::unique_lock<std::mutex>guard(mode_mutex_);success=false;const auto current=control_mode_.load();if(target>kModeManual){message="invalid control mode";actual=current;return;}if(target==kModeDisabled){disable_in_progress_.store(true);success=transact_enable(false,message);if(success){control_mode_.store(kModeDisabled);standby_active_.store(false);standby_condition_.notify_all();have_policy_.store(false);have_policy_source_.store(false);have_maintenance_source_.store(false);}else{control_mode_.store(kModeFault);standby_active_.store(false);standby_condition_.notify_all();have_policy_.store(false);queue_disable_no_wait();}disable_in_progress_.store(false);actual=control_mode_.load();return;}if(!profile_.control_allowed||!profile_.calibrated){message="Robot Profile is read-only or not calibrated";actual=current;return;}StateEnvelope state;std::string reason;if(!state_is_ready(state,reason)){message=reason;actual=current;return;}if(current==kModeDisabled&&lower_control_state_.load()!=2){message="lower controller must confirm DISABLED before enable";actual=current;return;}if(target==kModePolicy&&current!=kModeDisabled&&current!=kModeStandby){message="POLICY requires DISABLED or STANDBY";actual=current;return;}if((target==kModeSweep||target==kModeManual)&&current!=kModeDisabled){message="maintenance modes require DISABLED";actual=current;return;}if(target==kModeStandby&&current!=kModeDisabled&&current!=kModePolicy&&current!=kModeStandby){message="STANDBY requires DISABLED or POLICY";actual=current;return;}if(target==kModePolicy){have_policy_.store(false);have_policy_source_.store(false);if(current==kModeDisabled&&!transact_enable(true,message)){control_mode_.store(kModeFault);actual=kModeFault;return;}control_mode_.store(kModePolicy);success=true;message=current==kModeDisabled?"POLICY enabled without a hold/default-pose command; waiting for the first fresh policy command":"POLICY selected; waiting for a command from a new lower-state sequence";actual=target;return;}if(target==kModeSweep||target==kModeManual){capture_active_entry_positions(state);have_maintenance_source_.store(false);}
    auto hold=hold_current(state);if(current==kModeDisabled&&!transact_enable(true,message)){control_mode_.store(kModeFault);actual=kModeFault;return;}control_mode_.store(target);begin_output_arming_grace();queue_command(target,state.state.state_sample_seq,hold,monotonic_ns());if(target==kModeStandby){standby_start_=SteadyClock::now();standby_initial_=hold;standby_active_.store(true);std::unique_lock<std::mutex>wait_lock(standby_mutex_);const auto timeout=std::chrono::duration<double>(standby_ramp_s_+2.0);if(!standby_condition_.wait_for(wait_lock,timeout,[this]{return !standby_active_.load()||control_mode_.load()!=kModeStandby;})){message="STANDBY ramp completion timeout";control_mode_.store(kModeFault);std::string ignored;transact_enable(false,ignored);actual=kModeFault;return;}if(control_mode_.load()!=kModeStandby){message="STANDBY interrupted by link or control fault";actual=control_mode_.load();return;}}success=true;message=target==kModeStandby?"full-device STANDBY ramp complete":"full-device maintenance mode enabled; lower watchdog owns command freshness";actual=target;}
  void advance_standby(const StateEnvelope&e){if(control_mode_.load()!=kModeStandby)return;auto command=hold_current(e);const auto elapsed=std::chrono::duration<double>(SteadyClock::now()-standby_start_).count();const float alpha=standby_active_.load()?std::clamp(static_cast<float>(elapsed/standby_ramp_s_),0.0F,1.0F):1.0F;for(std::size_t i=0;i<command.count;++i){const auto id=command.ports[i].port_id;const auto*initial=find_command(standby_initial_,id);const auto*actuator=find_actuator(profile_,id);if(!initial||!actuator)continue;if(actuator->policy_owner==PolicyOwner::POLICY){const float goal=actuator->direction*(actuator->default_position_rad+actuator->mechanical_zero_rad);command.ports[i].position=initial->position+(goal-initial->position)*alpha;}else command.ports[i].position=initial->position;}const bool completed=standby_active_.load()&&alpha>=1;if(completed)standby_active_.store(false);queue_command(kModeStandby,e.state.state_sample_seq,command,e.receive_ns);if(completed)standby_condition_.notify_all();}
  void cache_auxiliary_contribution(const esd_link_msgs::msg::ActuatorCommand&msg){std::lock_guard<std::mutex>lock(auxiliary_mutex_);latest_auxiliary_contribution_=msg;}
  // Track the latest trustworthy physical pose until POLICY freezes the RC target.
  // A newly split CAN path may publish a zero placeholder while its port is offline.
  void advance_auxiliary_targets(const esd_link_msgs::msg::RobotState&state){std::lock_guard<std::mutex>lock(auxiliary_mutex_);const auto mode=control_mode_.load();for(std::size_t i=0;i<state.port_id.size();++i){const auto*actuator=find_actuator(profile_,state.port_id[i]);if(!actuator||actuator->policy_owner!=PolicyOwner::AUXILIARY)continue;const auto bit=1U<<actuator->port_id;const bool feedback_valid=i<state.valid_mask.size()&&(state.valid_mask[i]&0x03U)==0x03U&&(state.offline_port_mask&bit)==0U;if(!feedback_valid)continue;if((auxiliary_initialized_mask_&bit)==0U||mode!=kModePolicy){auxiliary_targets_[actuator->port_id]=state.position_rad[i];auxiliary_initialized_mask_|=bit;}}if(!wing_rc_enabled_.load()||mode!=kModePolicy)return;const auto now_ns=monotonic_ns();if(timestamp_is_stale(now_ns,last_joy_ns_.load(),500000000ULL))return;const float step=wing_velocity_.load()/static_cast<float>(expected_rate_);for(const auto&actuator:profile_.actuators){if(actuator.policy_owner!=PolicyOwner::AUXILIARY||std::abs(actuator.auxiliary_remote_direction)<1e-6F)continue;auto&target=auxiliary_targets_[actuator.port_id];target+=step*actuator.auxiliary_remote_direction;if(actuator.has_position_limits)target=std::clamp(target,actuator.position_min_rad,actuator.position_max_rad);}}
  esd_link_msgs::msg::ActuatorCommand internal_auxiliary_contribution(const esd_link_msgs::msg::RobotState&state,std::uint32_t source_state_sample_seq){esd_link_msgs::msg::ActuatorCommand command;command.header=state.header;command.owner=command.OWNER_AUXILIARY;command.mode=command.MODE_POLICY;command.profile_id=profile_.profile_id;command.profile_hash=profile_.profile_hash;command.source_state_sample_seq=source_state_sample_seq;command.generated_monotonic_ns=monotonic_ns();std::lock_guard<std::mutex>lock(auxiliary_mutex_);for(const auto&actuator:profile_.actuators){if(actuator.policy_owner!=PolicyOwner::AUXILIARY)continue;const auto bit=1U<<actuator.port_id;if((auxiliary_initialized_mask_&bit)==0U)continue;command.port_id.push_back(actuator.port_id);command.position_rad.push_back(auxiliary_targets_[actuator.port_id]);command.velocity_rad_s.push_back(0.0F);command.kp.push_back(actuator.kp);command.kd.push_back(actuator.kd);command.effort_nm.push_back(0.0F);}return command;}
  esd_link_msgs::msg::ActuatorCommand internal_auxiliary_contribution(const esd_link_msgs::msg::RobotState&state){return internal_auxiliary_contribution(state,state.state_sample_seq);}
  void actuator_contribution(const esd_link_msgs::msg::ActuatorCommand&msg){const auto mode=control_mode_.load();if(msg.mode!=mode||(mode!=kModePolicy&&mode!=kModeSweep&&mode!=kModeManual)){rejected_commands_.fetch_add(1);return;}esd_link_msgs::msg::RobotState state;{std::lock_guard<std::mutex>lock(robot_state_mutex_);if(!have_robot_state_){rejected_commands_.fetch_add(1);return;}state=latest_robot_state_;}std::vector<esd_link_msgs::msg::ActuatorCommand> contributions{msg};if(mode==kModePolicy){bool has_auxiliary=false;for(const auto&actuator:profile_.actuators)has_auxiliary=has_auxiliary||actuator.policy_owner==PolicyOwner::AUXILIARY;if(has_auxiliary){std::optional<esd_link_msgs::msg::ActuatorCommand> external;{std::lock_guard<std::mutex>lock(auxiliary_mutex_);if(latest_auxiliary_contribution_&&latest_auxiliary_contribution_->source_state_sample_seq==state.state_sample_seq)external=latest_auxiliary_contribution_;}contributions.push_back(external?*external:internal_auxiliary_contribution(state));}}esd_link_msgs::msg::ActuatorCommand final_command;std::string reason;if(!command_arbiter_->compose(mode,state,contributions,final_command,reason)){rejected_commands_.fetch_add(1);RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,"Command Arbiter rejected: %s",reason.c_str());return;}CompleteLowerCommand lower;if(!command_arbiter_->to_lower(final_command,lower,reason)){rejected_commands_.fetch_add(1);return;}auto&last=mode==kModePolicy?last_policy_source_:last_maintenance_source_;auto&have=mode==kModePolicy?have_policy_source_:have_maintenance_source_;if(!accept_new_source_sequence(msg.source_state_sample_seq,last,have))return;QueuedCommand queued;queued.count=lower.count;queued.ports=lower.ports;final_command.generated_monotonic_ns=monotonic_ns();final_command_pub_->publish(final_command);if(mode==kModePolicy)have_policy_.store(true);queue_command(mode,msg.source_state_sample_seq,queued,final_command.generated_monotonic_ns);}
  void joy_command(const sensor_msgs::msg::Joy&msg){const auto pressed=[&msg](int index){return static_cast<std::size_t>(index)<msg.buttons.size()&&msg.buttons[static_cast<std::size_t>(index)]==1;};wing_velocity_.store(wing_rc::velocity_from_buttons(pressed(wing_rc::kLbButton),pressed(wing_rc::kRbButton)));last_joy_ns_.store(monotonic_ns());}

  void native_set_zeros(const esd_link_msgs::srv::SetLowerZeros::Request&req,esd_link_msgs::srv::SetLowerZeros::Response&res){std::lock_guard<std::mutex>guard(mode_mutex_);if(control_mode_.load()!=kModeDisabled||lower_control_state_.load()!=2){res.success=false;res.message="SET_ZERO requires upper and lower DISABLED";return;}if(req.port_ids.empty()||req.port_ids.size()!=req.assigned_position_rad.size()||req.port_ids.size()>protocol::kMaxPortCount){res.success=false;res.message="invalid zero assignment lengths";return;}protocol::Frame frame;const auto tx=++transaction_sequence_;if(!protocol::build_set_zero(session_id_.load(),tx,req.persist,req.port_ids.data(),req.assigned_position_rad.data(),req.port_ids.size(),frame)){res.success=false;res.message="invalid zero assignments";return;}auto pending=submit_transaction(frame,tx,std::chrono::seconds(10));if(!pending->done){res.success=false;res.message="SET_ZERO timeout";return;}res.config_fingerprint=pending->result.config_fingerprint;res.success=pending->result.status==0;res.message=res.success?"zeros stored; identity recorded and session will be reopened":std::string("SET_ZERO rejected: ")+protocol::status_name(pending->result.status);if(res.success){verified_fingerprint_.store(res.config_fingerprint);std::string persist_error;if(!persist_identity_record(res.config_fingerprint,persist_error))res.message="zeros stored, but host identity record failed: "+persist_error;session_ready_.store(false);reopen_session_.store(true);wake_io();}}
  void set_gains(const motors::srv::SetMotorGains::Request&req,motors::srv::SetMotorGains::Response&res){std::lock_guard<std::mutex>lock(gain_mutex_);if(req.restore_defaults){kp_=default_kp_;kd_=default_kd_;res.success=true;res.message="restored ESD-Link startup gains";return;}if(req.kp.size()!=req.kd.size()||req.kp.empty()){res.success=false;res.message="kp/kd size mismatch";return;}std::vector<int>ids;if(req.motor_ids.empty()){if(req.kp.size()!=6){res.success=false;res.message="empty motor_ids requires six gains";return;}ids={1,2,3,4,5,6};}else{for(auto id:req.motor_ids)ids.push_back(id);}if(ids.size()!=req.kp.size()){res.success=false;res.message="motor_ids/kp/kd size mismatch";return;}for(std::size_t i=0;i<ids.size();++i){if(ids[i]<1||ids[i]>6||!std::isfinite(req.kp[i])||!std::isfinite(req.kd[i])||req.kp[i]<0||req.kd[i]<0){res.success=false;res.message="invalid gain assignment";return;}kp_[ids[i]-1]=req.kp[i];kd_[ids[i]-1]=req.kd[i];}res.success=true;res.message="ESD-Link runtime gains updated";}
  void read_motors(motors::srv::ReadMotors::Response&res){StateEnvelope state;std::string reason;if(!state_is_ready(state,reason)){res.success=false;res.message=reason;return;}std::ostringstream out;for(std::size_t i=0;i<motor_table_.count;++i){const auto id=motor_table_.entries[i].port_id;const auto*p=find_port(state.state,id);if(!p)continue;if(id>=1&&id<=6){const auto index=static_cast<std::size_t>(id-1);auto position=coordinates::policy_position_from_lower(p->position,sign_[index],defaults_[index]);if(velocity_only_[index])position=coordinates::wrap_to_limits(position,limits_[2*index],limits_[2*index+1]);out<<"P"<<static_cast<int>(id)<<" q="<<position<<" dq="<<coordinates::policy_direction_from_lower(p->velocity,sign_[index])<<" tau="<<coordinates::policy_direction_from_lower(p->effort,sign_[index]);}else if(id==7||id==8){const auto wing=static_cast<std::size_t>(id-7);out<<"P"<<static_cast<int>(id)<<" q="<<coordinates::policy_direction_from_lower(p->position,wing_sign_[wing])<<" dq="<<coordinates::policy_direction_from_lower(p->velocity,wing_sign_[wing])<<" tau="<<coordinates::policy_direction_from_lower(p->effort,wing_sign_[wing]);}else{out<<"P"<<static_cast<int>(id)<<" q="<<p->position<<" dq="<<p->velocity<<" tau="<<p->effort;}if(i+1<motor_table_.count)out<<"; ";}res.success=true;res.message=out.str();}
  void manual_control(const motors::srv::ControlMotor::Request&req,motors::srv::ControlMotor::Response&res){if(control_mode_.load()!=kModeManual){res.success=false;res.message="control_motor requires MANUAL_TEST mode";return;}if(!protocol::port_bit(expected_mask_,static_cast<std::uint8_t>(req.motor_id))){res.success=false;res.message="motor_id is not in the active motor table";return;}StateEnvelope state;std::string reason;if(!state_is_ready(state,reason)){res.success=false;res.message=reason;return;}auto command=hold_current(state);auto*port=find_command(command,static_cast<std::uint8_t>(req.motor_id));if(!port){res.success=false;res.message="motor_id is not in the active motor table";return;}if(req.motor_id<=6){const auto index=static_cast<std::size_t>(req.motor_id-1);if((velocity_only_[index]&&std::abs(req.velocity)>wheel_velocity_limits_[index==2?0:1])||(!velocity_only_[index]&&(req.position<limits_[2*index]||req.position>limits_[2*index+1]))){res.success=false;res.message="manual target outside configured limit";return;}port->position=coordinates::lower_position_from_policy(req.position,sign_[index],defaults_[index]);port->velocity=coordinates::lower_direction_from_policy(req.velocity,sign_[index]);port->effort=coordinates::lower_direction_from_policy(req.effort,sign_[index]);}else if(req.motor_id==7||req.motor_id==8){const auto wing=static_cast<std::size_t>(req.motor_id-7);if(req.position<wing_limits_[2*wing]||req.position>wing_limits_[2*wing+1]){res.success=false;res.message="manual wing target outside configured limit";return;}port->position=coordinates::lower_direction_from_policy(req.position,wing_sign_[wing]);port->velocity=coordinates::lower_direction_from_policy(req.velocity,wing_sign_[wing]);port->effort=coordinates::lower_direction_from_policy(req.effort,wing_sign_[wing]);}else{port->position=req.position;port->velocity=req.velocity;port->effort=req.effort;}if(!accept_new_source_sequence(state.state.state_sample_seq,last_maintenance_source_,have_maintenance_source_)||!queue_command(kModeManual,state.state.state_sample_seq,command,monotonic_ns())){res.success=false;res.message="manual command invalid or does not use a new lower-state sequence";return;}res.success=true;res.message="full configured-port command queued once; only selected target changed";}

  void publish_link_status(){esd_link_msgs::msg::LinkStatus msg;msg.header.stamp=now();msg.control_mode=control_mode_.load();msg.link_state=!connected_.load()?msg.DISCONNECTED:(!session_ready_.load()?msg.CONNECTING:(msg.control_mode==kModeDisabled?msg.READ_ONLY:(msg.control_mode==kModeFault?msg.FAULT:msg.READY)));msg.device_control_state=device_control_state_.load();msg.session_valid=session_ready_.load();msg.session_id=session_id_.load();msg.schema_id=schema_id_;msg.config_fingerprint=fingerprint_.load();msg.active_port_mask=expected_mask_;msg.offline_port_mask=offline_mask_.load();msg.fault_flags=fault_flags_.load();msg.last_reject_code=last_reject_code_.load();const auto now_ns=monotonic_ns(),last=last_state_ns_.load();msg.latest_state_age_ms=last?static_cast<float>(elapsed_ns_saturated(now_ns,last))/1e6F:std::numeric_limits<float>::infinity();const auto count=state_count_.load();const auto previous=last_status_state_count_.exchange(count);msg.state_rate_hz=static_cast<float>(count-previous);msg.maximum_interarrival_ms=static_cast<float>(max_gap_ns_.exchange(0))/1e6F;msg.valid_frames=decoder_.frames();msg.cobs_errors=decoder_.cobs_errors();msg.crc_errors=decoder_.crc_errors();msg.sequence_gaps=sequence_gaps_.load();msg.rejected_commands=rejected_commands_.load()+device_rejected_commands_.load();msg.watchdog_events=watchdog_events_.load();msg.control_fault_events=control_fault_events_.load();msg.reconnects=reconnects_.load();msg.device_valid_commands=device_valid_commands_.load();msg.device_invalid_frames=device_invalid_frames_.load();msg.device_rejected_commands=device_rejected_commands_.load();msg.device_command_age_ms=device_command_age_ms_.load();msg.last_command_sequence=last_sent_command_.load();msg.last_applied_command_sequence=last_applied_sequence_.load();msg.command_to_applied_ms=static_cast<float>(last_apply_latency_ns_.load())/1e6F;msg.observation_to_command_ms=static_cast<float>(last_observation_to_command_ns_.load())/1e6F;if(!msg.session_valid)msg.message="ESD-Link session unavailable";else if(msg.fault_flags||msg.offline_port_mask){std::ostringstream reason;reason<<"ESD-Link lower fault=0x"<<std::hex<<msg.fault_flags<<" offline=0x"<<msg.offline_port_mask;msg.message=reason.str();}else msg.message="ESD-Link session ready; timing fields are diagnostic only";link_pub_->publish(msg);}

  protocol::StreamDecoder decoder_;std::string serial_device_,serial_device_match_,serial_by_id_directory_,active_serial_device_,last_serial_error_,identity_record_path_,robot_profile_path_;RobotProfile profile_{};MotorTable motor_table_{};std::unique_ptr<RobotStateMapper>state_mapper_;std::unique_ptr<CommandArbiter>command_arbiter_;std::uint16_t layout_id_{1},schema_id_{1},expected_rate_{500};std::uint32_t expected_mask_{0x1FE};int rt_priority_{70},cpu_affinity_{-1},enable_confirm_timeout_ms_{1000},disable_confirm_timeout_ms_{6500};double standby_ramp_s_{3};
  std::array<float,6>defaults_{},sign_{},kp_{},kd_{},default_kp_{},default_kd_{};std::array<bool,6>velocity_only_{};std::array<float,12>limits_{};std::array<float,2>wheel_velocity_limits_{},wing_sign_{};std::array<float,4>wing_limits_{};std::array<float,32>active_entry_positions_{};
  int fd_{-1},wake_fd_{-1};std::atomic<bool>running_{false},connected_{false},session_ready_{false},reopen_session_{false},disable_in_progress_{false},config_in_flight_{false},config_busy_seen_{false};bool enforce_config_fingerprint_{true};std::thread io_thread_,publish_thread_;std::uint32_t host_nonce_{0},config_transaction_id_{0},open_attempts_{0};std::atomic<std::uint32_t>session_id_{0},fingerprint_{0},verified_fingerprint_{0},lower_boot_id_{0},tx_sequence_{0},transaction_sequence_{0};protocol::WireBuffer config_wire_{};int config_retries_{0};SteadyClock::time_point config_retry_at_{};
  LatestMailbox<StateEnvelope>state_mailbox_;std::mutex publish_mutex_;std::condition_variable publish_condition_;std::mutex cache_mutex_,control_state_mutex_,standby_mutex_;std::condition_variable control_state_condition_,standby_condition_;StateEnvelope latest_state_{};bool have_state_{false};
  std::mutex outbound_mutex_,pending_mutex_;std::deque<std::shared_ptr<PendingTransaction>>transactions_;std::vector<std::shared_ptr<PendingTransaction>>pending_;std::optional<QueuedCommand>queued_command_;
  std::atomic<std::uint8_t>control_mode_{kModeDisabled};std::atomic<std::uint64_t>output_arming_grace_until_ns_{0};std::mutex mode_mutex_,gain_mutex_,robot_state_mutex_,auxiliary_mutex_;std::atomic<bool>standby_active_{false},wing_rc_enabled_{false},have_policy_{false},have_policy_source_{false},have_maintenance_source_{false};SteadyClock::time_point standby_start_;QueuedCommand standby_initial_{};bool have_robot_state_{false};esd_link_msgs::msg::RobotState latest_robot_state_{};std::optional<esd_link_msgs::msg::ActuatorCommand>latest_auxiliary_contribution_;std::array<float,32>auxiliary_targets_{};std::uint32_t auxiliary_initialized_mask_{0};std::atomic<float>wing_velocity_{0};std::atomic<std::uint64_t>last_joy_ns_{0};std::atomic<std::uint32_t>last_policy_source_{0},last_maintenance_source_{0};
  std::atomic<std::uint64_t>state_count_{0},last_status_state_count_{0},last_state_ns_{0},max_gap_ns_{0},sequence_gaps_{0},rejected_commands_{0},reconnects_{0},watchdog_events_{0},control_fault_events_{0},last_apply_latency_ns_{0},last_observation_to_command_ns_{0},control_state_generation_{0};std::atomic<std::uint32_t>last_state_sequence_{0},last_sent_command_{0},last_applied_sequence_{0},device_valid_commands_{0},device_invalid_frames_{0},device_rejected_commands_{0},fault_flags_{0},offline_mask_{0};std::atomic<std::uint16_t>last_reject_code_{0},device_command_age_ms_{0};std::atomic<std::uint8_t>device_control_state_{0},lower_control_state_{0};std::array<AckEntry,256>acks_{};std::array<StateTime,256>state_times_{};std::mutex ack_mutex_;
  rclcpp::Publisher<esd_link_msgs::msg::LowerState>::SharedPtr lower_pub_;rclcpp::Publisher<esd_link_msgs::msg::RobotState>::SharedPtr robot_pub_;rclcpp::Publisher<esd_link_msgs::msg::ActuatorCommand>::SharedPtr final_command_pub_;rclcpp::Publisher<esd_link_msgs::msg::LinkStatus>::SharedPtr link_pub_;
  rclcpp::Subscription<esd_link_msgs::msg::ActuatorCommand>::SharedPtr policy_actuator_sub_,auxiliary_actuator_sub_,maintenance_actuator_sub_;rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Service<esd_link_msgs::srv::SetControlMode>::SharedPtr control_mode_service_;rclcpp::Service<esd_link_msgs::srv::SetLowerZeros>::SharedPtr zero_native_service_;rclcpp::Service<motors::srv::SetOperationalState>::SharedPtr operational_service_;rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr soft_disarm_service_;rclcpp::Service<motors::srv::SetZeros>::SharedPtr zero_compat_service_;rclcpp::Service<motors::srv::SetMotorGains>::SharedPtr gains_service_;rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr wing_rc_service_;rclcpp::Service<motors::srv::ReadMotors>::SharedPtr read_service_;rclcpp::Service<motors::srv::ControlMotor>::SharedPtr control_service_;rclcpp::Service<motors::srv::ResetMotors>::SharedPtr unsupported_reset_;rclcpp::Service<motors::srv::ClearErrors>::SharedPtr unsupported_clear_;rclcpp::Service<motors::srv::IdentifyMotorID>::SharedPtr unsupported_identify_;rclcpp::Service<motors::srv::SetMasterID>::SharedPtr unsupported_master_id_;rclcpp::Service<motors::srv::ScanMotors>::SharedPtr unsupported_scan_;rclcpp::Service<motors::srv::RunSafetyCheck>::SharedPtr unsupported_safety_;rclcpp::TimerBase::SharedPtr diagnostics_timer_;
};

}  // namespace esd_link_bridge

int main(int argc,char**argv){rclcpp::init(argc,argv);try{auto node=std::make_shared<esd_link_bridge::EsdLinkBridgeNode>();rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),4);executor.add_node(node);executor.spin();}catch(const std::exception&e){std::fprintf(stderr,"esd_link_bridge fatal: %s\n",e.what());rclcpp::shutdown();return 1;}rclcpp::shutdown();return 0;}
