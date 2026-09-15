#include "esd_link_bridge/protocol.hpp"
#include "esd_link_bridge/latest_mailbox.hpp"
#include "esd_link_bridge/coordinates.hpp"
#include "esd_link_bridge/time_utils.hpp"

#include <esd_link_msgs/msg/link_status.hpp>
#include <esd_link_msgs/msg/lower_state.hpp>
#include <esd_link_msgs/msg/maintenance_command.hpp>
#include <esd_link_msgs/msg/policy_command.hpp>
#include <esd_link_msgs/msg/wing_command.hpp>
#include <esd_link_msgs/srv/set_control_mode.hpp>
#include <esd_link_msgs/srv/set_lower_zeros.hpp>
#include <motors/msg/motor_runtime_status.hpp>
#include <motors/msg/wing_runtime_status.hpp>
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
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/eventfd.h>
#include <sys/file.h>
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
  std::array<ActuatorCommand, 8> ports{};
};

constexpr std::uint8_t kModeDisabled = 0;
constexpr std::uint8_t kModeStandby = 1;
constexpr std::uint8_t kModePolicy = 2;
constexpr std::uint8_t kModeSweep = 3;
constexpr std::uint8_t kModeManual = 4;
constexpr std::uint8_t kModeFault = 5;

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
    RCLCPP_INFO(get_logger(), "ESD-Link bridge started read-only: device=%s, auto_enable=false",
                serial_device_.c_str());
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
    declare_parameter<std::string>("serial_device", "");
    declare_parameter<int>("layout_id", 1); declare_parameter<int>("schema_id", 1);
    declare_parameter<int64_t>("expected_active_port_mask", 0x1FE);
    declare_parameter<int>("expected_control_rate_hz", 250);
    declare_parameter<int64_t>("expected_config_fingerprint", 0);
    declare_parameter<std::string>("identity_record_path", "");
    declare_parameter<int>("realtime_priority", 70); declare_parameter<int>("cpu_affinity", -1);
    declare_parameter<int>("enable_confirm_timeout_ms", 1000);
    declare_parameter<int>("disable_confirm_timeout_ms", 6500);
    declare_parameter<double>("standby_ramp_seconds", 3.0);
    declare_parameter<std::vector<int64_t>>("motor_ids", {1,2,3,4,5,6});
    declare_parameter<std::vector<double>>("joint_default_angle", {-0.92020,0.98338,0,-0.92020,0.98338,0});
    declare_parameter<std::vector<bool>>("flipped_motors", {true,false,true,false,true,false});
    declare_parameter<std::vector<bool>>("velocity_only", {false,false,true,false,false,true});
    declare_parameter<std::vector<double>>("kp", {2,8,0,2,8,0});
    declare_parameter<std::vector<double>>("kd", {0.1,0.8,0.2,0.1,0.8,0.2});
    declare_parameter<std::vector<double>>("joint_position_limits", {-0.89,0.84,-0.89,0.24,-6.28,6.28,-0.88,0.84,-0.89,0.22,-6.28,6.28});
    declare_parameter<std::vector<double>>("wheel_velocity_limits", {35.0,35.0});
    declare_parameter<std::vector<double>>("wing_signs", {-1.0,-1.0});
    declare_parameter<std::vector<double>>("wing_position_limits", {-0.7,2.0,-2.0,0.7});
    declare_parameter<std::vector<double>>("wing_kp", {10,10}); declare_parameter<std::vector<double>>("wing_kd", {0.5,0.5});
    declare_parameter<bool>("auto_enable", false);
  }

  template <std::size_t N>
  std::array<float, N> float_array_parameter(const std::string &name) {
    const auto values = get_parameter(name).as_double_array();
    if (values.size() != N) throw std::runtime_error(name + " must contain " + std::to_string(N) + " values");
    std::array<float, N> out{}; std::transform(values.begin(), values.end(), out.begin(), [](double v){return static_cast<float>(v);}); return out;
  }
  void load_parameters() {
    serial_device_=get_parameter("serial_device").as_string(); if(serial_device_.empty())throw std::runtime_error("serial_device is required");
    layout_id_=static_cast<std::uint16_t>(get_parameter("layout_id").as_int()); schema_id_=static_cast<std::uint16_t>(get_parameter("schema_id").as_int());
    expected_mask_=static_cast<std::uint32_t>(get_parameter("expected_active_port_mask").as_int()); expected_rate_=static_cast<std::uint16_t>(get_parameter("expected_control_rate_hz").as_int());
    expected_fingerprint_.store(static_cast<std::uint32_t>(get_parameter("expected_config_fingerprint").as_int()));identity_record_path_=get_parameter("identity_record_path").as_string();if(!identity_record_path_.empty())load_identity_record();
    rt_priority_=get_parameter("realtime_priority").as_int();cpu_affinity_=get_parameter("cpu_affinity").as_int();enable_confirm_timeout_ms_=get_parameter("enable_confirm_timeout_ms").as_int();disable_confirm_timeout_ms_=get_parameter("disable_confirm_timeout_ms").as_int();standby_ramp_s_=get_parameter("standby_ramp_seconds").as_double();
    defaults_=float_array_parameter<6>("joint_default_angle");kp_=float_array_parameter<6>("kp");kd_=float_array_parameter<6>("kd");limits_=float_array_parameter<12>("joint_position_limits");wheel_velocity_limits_=float_array_parameter<2>("wheel_velocity_limits");wing_sign_=float_array_parameter<2>("wing_signs");wing_limits_=float_array_parameter<4>("wing_position_limits");wing_kp_=float_array_parameter<2>("wing_kp");wing_kd_=float_array_parameter<2>("wing_kd");
    const auto flips=get_parameter("flipped_motors").as_bool_array();const auto velocity=get_parameter("velocity_only").as_bool_array();if(flips.size()!=6||velocity.size()!=6)throw std::runtime_error("flipped_motors and velocity_only must contain 6 values");for(std::size_t i=0;i<6;++i){sign_[i]=flips[i]?-1.0F:1.0F;velocity_only_[i]=velocity[i];default_kp_[i]=kp_[i];default_kd_[i]=kd_[i];}
    for(auto&wing_sign:wing_sign_){if(!std::isfinite(wing_sign)||std::abs(wing_sign)<1e-6F)throw std::runtime_error("wing_signs entries must be finite and non-zero");wing_sign=wing_sign<0?-1.0F:1.0F;}
    for(std::size_t i=0;i<6;++i){if(!std::isfinite(limits_[2*i])||!std::isfinite(limits_[2*i+1])||limits_[2*i]>=limits_[2*i+1])throw std::runtime_error("joint_position_limits entries must be finite ordered pairs");}
    for(std::size_t i=0;i<2;++i){if(!std::isfinite(wing_limits_[2*i])||!std::isfinite(wing_limits_[2*i+1])||wing_limits_[2*i]>=wing_limits_[2*i+1])throw std::runtime_error("wing_position_limits entries must be finite ordered pairs");}
    if(get_parameter("auto_enable").as_bool())throw std::runtime_error("auto_enable is forbidden for frozen ESD-Link hardware");
  }

  void create_ros_interfaces() {
    auto sensor_qos=rclcpp::SensorDataQoS();auto latched=rclcpp::QoS(1).reliable().transient_local();
    lower_pub_=create_publisher<esd_link_msgs::msg::LowerState>("/lower/state",sensor_qos);
    link_pub_=create_publisher<esd_link_msgs::msg::LinkStatus>("/lower/link_status",latched);
    imu_pub_=create_publisher<sensor_msgs::msg::Imu>("/IMU_data",sensor_qos);
    joints_pub_=create_publisher<sensor_msgs::msg::JointState>("/policy/joint_states",sensor_qos);
    wings_pub_=create_publisher<sensor_msgs::msg::JointState>("/policy/wing_angles",sensor_qos);
    motor_status_pub_=create_publisher<motors::msg::MotorRuntimeStatus>("/motors/runtime_status",latched);
    wing_status_pub_=create_publisher<motors::msg::WingRuntimeStatus>("/wing/runtime_status",latched);
    policy_sub_=create_subscription<esd_link_msgs::msg::PolicyCommand>("/policy/stamped_commands",rclcpp::QoS(1).reliable(),[this](esd_link_msgs::msg::PolicyCommand::SharedPtr msg){policy_command(*msg);});
    maintenance_sub_=create_subscription<esd_link_msgs::msg::MaintenanceCommand>("/maintenance/commands",rclcpp::QoS(1).reliable(),[this](esd_link_msgs::msg::MaintenanceCommand::SharedPtr msg){maintenance_command(*msg);});
    wing_sub_=create_subscription<esd_link_msgs::msg::WingCommand>("/wing/commands",rclcpp::QoS(1).reliable(),[this](esd_link_msgs::msg::WingCommand::SharedPtr msg){wing_command(*msg);});
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

  void load_identity_record(){std::ifstream stream(identity_record_path_);std::string line;while(std::getline(stream,line)){constexpr const char*prefix="config_fingerprint=";if(line.rfind(prefix,0)==0){try{expected_fingerprint_.store(static_cast<std::uint32_t>(std::stoul(line.substr(std::strlen(prefix)),nullptr,0)));RCLCPP_INFO(get_logger(),"loaded ESD-Link config fingerprint from %s",identity_record_path_.c_str());}catch(const std::exception&){RCLCPP_WARN(get_logger(),"invalid ESD-Link identity record %s",identity_record_path_.c_str());}return;}}}
  bool persist_identity_record(std::uint32_t config_fingerprint,std::string&error){if(identity_record_path_.empty())return true;try{const std::filesystem::path path(identity_record_path_);if(path.has_parent_path())std::filesystem::create_directories(path.parent_path());const auto temporary=path.string()+".tmp";{std::ofstream stream(temporary,std::ios::trunc);if(!stream)throw std::runtime_error("cannot open temporary identity record");stream<<"schema_id="<<schema_id_<<"\nlayout_id="<<layout_id_<<"\nactive_port_mask=0x"<<std::hex<<std::setw(8)<<std::setfill('0')<<expected_mask_<<"\nconfig_fingerprint=0x"<<std::setw(8)<<config_fingerprint<<"\n";stream.flush();if(!stream)throw std::runtime_error("cannot flush temporary identity record");}std::filesystem::rename(temporary,path);return true;}catch(const std::exception&exception){error=exception.what();return false;}}

  void configure_realtime() {
    pthread_setname_np(pthread_self(),"esd_link_io");
    if(cpu_affinity_>=0){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu_affinity_,&set);const int result=pthread_setaffinity_np(pthread_self(),sizeof(set),&set);if(result!=0)std::fprintf(stderr,"esd_link_io: cannot set CPU affinity: %s\n",std::strerror(result));}
    if(rt_priority_>0){sched_param p{};p.sched_priority=rt_priority_;const int result=pthread_setschedparam(pthread_self(),SCHED_FIFO,&p);if(result!=0)std::fprintf(stderr,"esd_link_io: cannot set SCHED_FIFO %d: %s\n",rt_priority_,std::strerror(result));}
  }
  bool open_serial() {
    fd_=::open(serial_device_.c_str(),O_RDWR|O_NOCTTY|O_NONBLOCK|O_CLOEXEC);if(fd_<0)return false;
    if(::flock(fd_,LOCK_EX|LOCK_NB)!=0){::close(fd_);fd_=-1;return false;}
    termios tty{};if(tcgetattr(fd_,&tty)!=0){close_serial();return false;}cfmakeraw(&tty);cfsetispeed(&tty,B115200);cfsetospeed(&tty,B115200);tty.c_cflag|=CLOCAL|CREAD;tty.c_cflag&=~CRTSCTS;tty.c_cc[VMIN]=0;tty.c_cc[VTIME]=0;if(tcsetattr(fd_,TCSANOW,&tty)!=0){close_serial();return false;}tcflush(fd_,TCIOFLUSH);decoder_.reset();connected_.store(true);session_ready_.store(false);return true;
  }
  void close_serial(){if(fd_>=0){::flock(fd_,LOCK_UN);::close(fd_);fd_=-1;}connected_.store(false);session_ready_.store(false);session_id_.store(0);have_policy_.store(false);have_policy_source_.store(false);have_maintenance_source_.store(false);standby_active_.store(false);{std::lock_guard<std::mutex>lock(cache_mutex_);have_state_=false;}lower_control_state_.store(0);control_state_condition_.notify_all();standby_condition_.notify_all();if(control_mode_.exchange(kModeDisabled)!=kModeDisabled)std::fprintf(stderr,"esd_link_io: disconnected; control mode forced DISABLED\n");}
  bool write_wire(const protocol::WireBuffer&w){std::size_t off=0;while(off<w.size&&running_.load()){const auto n=::write(fd_,w.bytes.data()+off,w.size-off);if(n>0){off+=static_cast<std::size_t>(n);continue;}if(n<0&&(errno==EAGAIN||errno==EINTR)){pollfd p{fd_,POLLOUT,0};if(::poll(&p,1,10)>=0)continue;}return false;}return off==w.size;}
  bool send_frame(protocol::Frame frame){frame.sequence=++tx_sequence_;protocol::WireBuffer wire;if(!protocol::encode_frame(frame,wire))return false;return write_wire(wire);}
  void send_open(){std::random_device rd;host_nonce_=static_cast<std::uint32_t>(rd())^static_cast<std::uint32_t>(monotonic_ns());if(!host_nonce_)host_nonce_=1;protocol::Frame f;if(protocol::build_open_session(host_nonce_,layout_id_,schema_id_,expected_fingerprint_.load(),f))send_frame(f);}

  void io_loop(){configure_realtime();auto next_open=SteadyClock::now();auto next_handshake=SteadyClock::now();std::array<std::uint8_t,2048> read_buffer{};while(running_.load()){
      if(fd_<0){if(SteadyClock::now()>=next_open){if(open_serial()){reconnects_.fetch_add(1);send_open();next_handshake=SteadyClock::now()+std::chrono::seconds(1);std::fprintf(stderr,"esd_link_io: opened %s\n",serial_device_.c_str());}next_open=SteadyClock::now()+std::chrono::seconds(1);}std::this_thread::sleep_for(std::chrono::milliseconds(100));continue;}
      if(reopen_session_.exchange(false)||(!session_ready_.load()&&SteadyClock::now()>=next_handshake)){send_open();next_handshake=SteadyClock::now()+std::chrono::seconds(1);}
      pollfd fds[2]={{fd_,POLLIN|POLLERR|POLLHUP,0},{wake_fd_,POLLIN,0}};const int result=::poll(fds,2,100);
      if(result<0){if(errno==EINTR)continue;close_serial();continue;}
      if(fds[1].revents&POLLIN){std::uint64_t count;while(::read(wake_fd_,&count,sizeof(count))>0){}drain_outbound();}
      if(fds[0].revents&(POLLERR|POLLHUP|POLLNVAL)){close_serial();continue;}
      if(fds[0].revents&POLLIN){for(;;){const auto n=::read(fd_,read_buffer.data(),read_buffer.size());if(n>0){decoder_.feed(read_buffer.data(),static_cast<std::size_t>(n),monotonic_ns());continue;}if(n<0&&errno!=EAGAIN&&errno!=EINTR)close_serial();break;}}
    }}
  void drain_outbound(){std::deque<std::shared_ptr<PendingTransaction>> transactions;std::optional<QueuedCommand> command;{std::lock_guard<std::mutex> lock(outbound_mutex_);transactions.swap(transactions_);command.swap(queued_command_);}for(auto&t:transactions)if(fd_>=0&&session_ready_.load())send_frame(t->frame);if(command&&!disable_in_progress_.load()&&fd_>=0&&session_ready_.load()&&command->expected_mode==control_mode_.load()){protocol::Frame f;if(protocol::build_actuator_command(session_id_.load(),layout_id_,command->source_state_sequence,fingerprint_.load(),command->ports,f)){const auto seq=tx_sequence_.load()+1;if(send_frame(f)){const auto sent=monotonic_ns();std::lock_guard<std::mutex>lock(ack_mutex_);const auto&state_time=state_times_[command->source_state_sequence%state_times_.size()];const auto observation_ns=state_time.sequence==command->source_state_sequence?state_time.receive_ns:0;acks_[seq%acks_.size()]={seq,sent,observation_ns};if(observation_ns&&sent>=observation_ns)last_observation_to_command_ns_.store(sent-observation_ns);last_sent_command_.store(seq);}}}}

  void observe_lower_control_state(std::uint8_t state) {
    lower_control_state_.store(state);
    device_control_state_.store(state);
    control_state_generation_.fetch_add(1);
    control_state_condition_.notify_all();
    if (state == 6) {
      fault_from_io();
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

  void on_frame(const protocol::Frame&frame,std::uint64_t receive_ns){if(frame.kind==protocol::Kind::Response){protocol::SessionInfo session;if(protocol::parse_session_info(frame,session)){handle_session(session);return;}protocol::TransactionResult result;if(protocol::parse_transaction_result(frame,result)&&result.status!=0xFFFF){complete_transaction(result);return;}return;}
    protocol::RobotState state;if(protocol::parse_robot_state(frame,state)){if(!session_ready_.load()||state.session_id!=session_id_.load()||state.schema_id!=schema_id_)return;observe_lower_control_state(state.control_state);const auto parsed=monotonic_ns();StateEnvelope envelope;envelope.state=state;envelope.receive_ns=receive_ns;envelope.parse_done_ns=parsed;update_state_stats(state,receive_ns);state_mailbox_.publish(envelope);publish_condition_.notify_one();process_ack(state.last_applied_command_seq,receive_ns);return;}
    protocol::DeviceStatus device;if(protocol::parse_device_status(frame,device)&&device.session_id==session_id_.load()){observe_lower_control_state(device.control_state);device_valid_commands_.store(device.valid_command_count);device_invalid_frames_.store(device.invalid_frame_count);device_rejected_commands_.store(device.rejected_command_count);device_command_age_ms_.store(device.command_age_ms);last_reject_code_.store(device.last_reject_code);const auto previous_faults=fault_flags_.exchange(device.fault_flags);if((device.fault_flags&2U)&&!(previous_faults&2U))watchdog_events_.fetch_add(1);if((device.fault_flags&0x80U)&&!(previous_faults&0x80U))control_fault_events_.fetch_add(1);offline_mask_.store(device.offline_port_mask);if(device.fault_flags||device.offline_port_mask||device.control_state==6)fault_from_io();return;}}
  void handle_session(const protocol::SessionInfo&s){const auto expected_fingerprint=expected_fingerprint_.load();if(s.status||s.host_nonce!=host_nonce_||s.business_version!=1||s.layout_id!=layout_id_||s.schema_id!=schema_id_||s.active_port_mask!=expected_mask_||s.control_rate_hz!=expected_rate_||(expected_fingerprint&&s.config_fingerprint!=expected_fingerprint)){std::fprintf(stderr,"esd_link_io: session mismatch status=%s layout=%u schema=%u mask=0x%08x rate=%u\n",protocol::status_name(s.status),s.layout_id,s.schema_id,s.active_port_mask,s.control_rate_hz);session_ready_.store(false);return;}session_id_.store(s.session_id);fingerprint_.store(s.config_fingerprint);lower_boot_id_.store(s.lower_boot_id);lower_control_state_.store(s.control_state);device_control_state_.store(s.control_state);fault_flags_.store(0);offline_mask_.store(0);last_reject_code_.store(0);last_state_sequence_.store(0);last_state_ns_.store(0);max_gap_ns_.store(0);last_status_state_count_.store(state_count_.load());have_policy_source_.store(false);have_maintenance_source_.store(false);{std::lock_guard<std::mutex>lock(cache_mutex_);have_state_=false;}{std::lock_guard<std::mutex>lock(ack_mutex_);acks_.fill({});state_times_.fill({});}control_state_condition_.notify_all();session_ready_.store(true);std::fprintf(stderr,"esd_link_io: session ready session=%u boot=%u fingerprint=0x%08x\n",s.session_id,s.lower_boot_id,s.config_fingerprint);}
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
    esd_link_msgs::msg::LowerState msg;msg.header.stamp=now();msg.header.frame_id="base_link";msg.host_monotonic_ns=e.receive_ns;msg.parse_done_monotonic_ns=e.parse_done_ns;msg.session_id=e.state.session_id;msg.schema_id=e.state.schema_id;msg.device_sample_time_us=e.state.sample_time_us;msg.state_sample_seq=e.state.state_sample_seq;msg.last_applied_command_seq=e.state.last_applied_command_seq;msg.command_status_flags=e.state.command_status_flags;msg.control_state=e.state.control_state;msg.imu_valid_mask=e.state.imu_valid_mask;msg.fault_flags=fault_flags_.load();msg.active_port_mask=expected_mask_;msg.offline_port_mask=offline_mask_.load();
    msg.imu.header=msg.header;msg.imu.orientation.x=e.state.quaternion[0];msg.imu.orientation.y=e.state.quaternion[1];msg.imu.orientation.z=e.state.quaternion[2];msg.imu.orientation.w=e.state.quaternion[3];msg.imu.angular_velocity.x=-e.state.gyro[0];msg.imu.angular_velocity.y=-e.state.gyro[1];msg.imu.angular_velocity.z=e.state.gyro[2];msg.imu.linear_acceleration.x=-e.state.accel[0];msg.imu.linear_acceleration.y=-e.state.accel[1];msg.imu.linear_acceleration.z=e.state.accel[2];const auto projected=gravity(e.state.quaternion);msg.imu.angular_velocity_covariance[0]=-projected[0];msg.imu.angular_velocity_covariance[1]=-projected[1];msg.imu.angular_velocity_covariance[2]=projected[2];
    for(std::size_t i=0;i<8;++i){const auto id=e.state.ports[i].id;msg.port_id[i]=id;msg.valid_mask[i]=e.state.ports[i].valid_mask;if(id>=1&&id<=6){const auto index=static_cast<std::size_t>(id-1);auto position=coordinates::policy_position_from_lower(e.state.ports[i].position,sign_[index],defaults_[index]);if(velocity_only_[index])position=coordinates::wrap_to_limits(position,limits_[2*index],limits_[2*index+1]);msg.position_rad[i]=position;msg.velocity_rad_s[i]=coordinates::policy_direction_from_lower(e.state.ports[i].velocity,sign_[index]);msg.effort_nm[i]=coordinates::policy_direction_from_lower(e.state.ports[i].effort,sign_[index]);}else{const auto wing=static_cast<std::size_t>(id-7);msg.position_rad[i]=coordinates::policy_direction_from_lower(e.state.ports[i].position,wing_sign_[wing]);msg.velocity_rad_s[i]=coordinates::policy_direction_from_lower(e.state.ports[i].velocity,wing_sign_[wing]);msg.effort_nm[i]=coordinates::policy_direction_from_lower(e.state.ports[i].effort,wing_sign_[wing]);}}lower_pub_->publish(msg);publish_legacy(e,msg.header);publish_runtime_status(e,msg.header);advance_standby(e);advance_wing(e);}
  std::array<float,3> gravity(const std::array<float,4>&q){const float x=q[0],y=q[1],z=q[2],w=q[3];return {-2*(x*z-y*w),-2*(y*z+x*w),-(1-2*x*x-2*y*y)};}
  void publish_legacy(const StateEnvelope&e,const std_msgs::msg::Header&header){sensor_msgs::msg::Imu imu;imu.header=header;imu.orientation.x=e.state.quaternion[0];imu.orientation.y=e.state.quaternion[1];imu.orientation.z=e.state.quaternion[2];imu.orientation.w=e.state.quaternion[3];imu.angular_velocity.x=-e.state.gyro[0];imu.angular_velocity.y=-e.state.gyro[1];imu.angular_velocity.z=e.state.gyro[2];imu.linear_acceleration.x=-e.state.accel[0];imu.linear_acceleration.y=-e.state.accel[1];imu.linear_acceleration.z=e.state.accel[2];const auto g=gravity(e.state.quaternion);imu.angular_velocity_covariance[0]=-g[0];imu.angular_velocity_covariance[1]=-g[1];imu.angular_velocity_covariance[2]=g[2];imu_pub_->publish(imu);
    sensor_msgs::msg::JointState joints;joints.header=header;for(std::size_t i=0;i<6;++i){const auto*p=find_port(e.state,static_cast<std::uint8_t>(i+1));if(!p)continue;joints.name.push_back("motor_"+std::to_string(i+1));auto pos=coordinates::policy_position_from_lower(p->position,sign_[i],defaults_[i]);if(velocity_only_[i])pos=coordinates::wrap_to_limits(pos,limits_[2*i],limits_[2*i+1]);joints.position.push_back(pos);joints.velocity.push_back(coordinates::policy_direction_from_lower(p->velocity,sign_[i]));joints.effort.push_back(coordinates::policy_direction_from_lower(p->effort,sign_[i]));}joints_pub_->publish(joints);
    sensor_msgs::msg::JointState wings;wings.header=header;for(std::uint8_t id=7;id<=8;++id){const auto*p=find_port(e.state,id);if(p){const auto wing=static_cast<std::size_t>(id-7);wings.name.push_back("motor_"+std::to_string(id));wings.position.push_back(coordinates::policy_direction_from_lower(p->position,wing_sign_[wing]));wings.velocity.push_back(coordinates::policy_direction_from_lower(p->velocity,wing_sign_[wing]));wings.effort.push_back(coordinates::policy_direction_from_lower(p->effort,wing_sign_[wing]));}}wings_pub_->publish(wings);}
  const protocol::PortState*find_port(const protocol::RobotState&s,std::uint8_t id)const{for(std::size_t i=0;i<s.port_count;++i)if(s.ports[i].id==id)return&s.ports[i];return nullptr;}
  void publish_runtime_status(const StateEnvelope&e,const std_msgs::msg::Header&header){motors::msg::MotorRuntimeStatus motor;motor.stamp=header.stamp;const auto mode=control_mode_.load();motor.state=compat_motor_state(mode);motor.fault_code=mode==kModeFault?"ESD_LINK_FAULT":"";motor.message=session_ready_.load()?"ESD-Link":"session unavailable";for(std::uint8_t id=1;id<=6;++id){const auto*p=find_port(e.state,id);motor.motor_ids.push_back(id);motor.online.push_back(p&&p->valid_mask!=0);motor.position.push_back(p?coordinates::policy_position_from_lower(p->position,sign_[id-1],defaults_[id-1]):std::numeric_limits<float>::quiet_NaN());motor.velocity.push_back(p?coordinates::policy_direction_from_lower(p->velocity,sign_[id-1]):std::numeric_limits<float>::quiet_NaN());motor.effort.push_back(p?coordinates::policy_direction_from_lower(p->effort,sign_[id-1]):std::numeric_limits<float>::quiet_NaN());}motor_status_pub_->publish(motor);
    motors::msg::WingRuntimeStatus wing;wing.stamp=header.stamp;wing.state=wing_rc_enabled_.load()?2:1;wing.feedback_ok=true;wing.motor_ids={7,8};for(std::uint8_t id=7;id<=8;++id){const auto*p=find_port(e.state,id);const auto index=static_cast<std::size_t>(id-7);wing.position.push_back(p?coordinates::policy_direction_from_lower(p->position,wing_sign_[index]):std::numeric_limits<float>::quiet_NaN());wing.velocity.push_back(p?coordinates::policy_direction_from_lower(p->velocity,wing_sign_[index]):std::numeric_limits<float>::quiet_NaN());wing.feedback_ok=wing.feedback_ok&&p&&p->valid_mask!=0;}wing.message="ESD-Link";wing_status_pub_->publish(wing);}

  bool state_is_ready(StateEnvelope&state,std::string&reason){std::lock_guard<std::mutex>lock(cache_mutex_);if(!have_state_){reason="no lower state";return false;}state=latest_state_;if((state.state.imu_valid_mask&0x07U)!=0x07U){reason="IMU invalid";return false;}for(const auto value:state.state.gyro)if(!std::isfinite(value)){reason="IMU gyro non-finite";return false;}for(const auto value:state.state.quaternion)if(!std::isfinite(value)){reason="IMU quaternion non-finite";return false;}for(const auto value:state.state.accel)if(!std::isfinite(value)){reason="IMU acceleration non-finite";return false;}for(std::uint8_t id=1;id<=8;++id){const auto*p=find_port(state.state,id);if(!p||(p->valid_mask&0x03U)!=0x03U||!std::isfinite(p->position)||!std::isfinite(p->velocity)||!std::isfinite(p->effort)){reason="port "+std::to_string(id)+" invalid";return false;}}if(lower_control_state_.load()>=5||fault_flags_.load()||offline_mask_.load()){reason="lower controller fault/offline";return false;}return true;}
  void capture_active_entry_positions(const StateEnvelope&state){for(std::size_t i=0;i<6;++i){const auto*p=find_port(state.state,static_cast<std::uint8_t>(i+1));active_entry_positions_[i]=coordinates::policy_position_from_lower(p->position,sign_[i],defaults_[i]);}for(std::size_t i=0;i<2;++i){const auto*p=find_port(state.state,static_cast<std::uint8_t>(i+7));active_entry_positions_[i+6]=coordinates::policy_direction_from_lower(p->position,wing_sign_[i]);}}
  bool maintenance_position_target_is_safe(std::uint8_t id,float target,float low,float high)const{if(target>=low&&target<=high)return true;const auto entry=active_entry_positions_[id-1];constexpr float tolerance=0.01F;if(entry>high)return target>=high&&target<=entry+tolerance;if(entry<low)return target<=low&&target>=entry-tolerance;return false;}
  std::pair<std::array<float,6>,std::array<float,6>> gains(){std::lock_guard<std::mutex>lock(gain_mutex_);return {kp_,kd_};}
  std::array<ActuatorCommand,8> hold_current(const StateEnvelope&s){const auto current_gains=gains();std::array<float,2>wing_kp,wing_kd;{std::lock_guard<std::mutex>lock(wing_mutex_);wing_kp=wing_kp_;wing_kd=wing_kd_;}std::array<ActuatorCommand,8> out{};for(std::size_t i=0;i<8;++i){const auto*p=find_port(s.state,static_cast<std::uint8_t>(i+1));out[i].port_id=i+1;out[i].position=p?p->position:0;out[i].velocity=0;out[i].kp=i<6?current_gains.first[i]:wing_kp[i-6];out[i].kd=i<6?current_gains.second[i]:wing_kd[i-6];out[i].effort=0;}return out;}
  bool queue_command(std::uint8_t mode,std::uint32_t source,const std::array<ActuatorCommand,8>&ports,std::uint64_t origin){for(const auto&p:ports)if(!std::isfinite(p.position)||!std::isfinite(p.velocity)||!std::isfinite(p.kp)||!std::isfinite(p.kd)||!std::isfinite(p.effort)||p.kp<0||p.kd<0){rejected_commands_.fetch_add(1);return false;}{std::lock_guard<std::mutex>lock(outbound_mutex_);queued_command_=QueuedCommand{mode,source,origin,ports};}wake_io();return true;}
  void wake_io(){if(wake_fd_>=0){std::uint64_t one=1;(void)::write(wake_fd_,&one,sizeof(one));}}
  std::shared_ptr<PendingTransaction> submit_transaction(protocol::Frame frame,std::uint32_t transaction_id,std::chrono::milliseconds timeout){auto pending=std::make_shared<PendingTransaction>();pending->transaction_id=transaction_id;frame.sequence=0;pending->frame=frame;{std::lock_guard<std::mutex>lock(pending_mutex_);pending_.push_back(pending);}{std::lock_guard<std::mutex>lock(outbound_mutex_);transactions_.push_back(pending);}wake_io();std::unique_lock<std::mutex>lock(pending->mutex);if(!pending->condition.wait_for(lock,timeout,[&]{return pending->done;})){std::lock_guard<std::mutex>p_lock(pending_mutex_);pending_.erase(std::remove(pending_.begin(),pending_.end(),pending),pending_.end());}return pending;}
  bool transact_enable(bool enable,std::string&message){if(!session_ready_.load()){message="ESD-Link session unavailable";return false;}protocol::Frame frame;const auto tx=++transaction_sequence_;if(!protocol::build_set_enable(session_id_.load(),tx,enable,frame)){message="cannot build SET_ENABLE";return false;}auto pending=submit_transaction(frame,tx,std::chrono::seconds(5));if(!pending->done){message="SET_ENABLE timeout";return false;}if(pending->result.status){message=std::string("SET_ENABLE rejected: ")+protocol::status_name(pending->result.status);return false;}const auto confirmation_generation=control_state_generation_.load();std::unique_lock<std::mutex>lock(control_state_mutex_);const auto confirm_timeout=std::chrono::milliseconds(enable?enable_confirm_timeout_ms_:disable_confirm_timeout_ms_);const bool confirmed=control_state_condition_.wait_for(lock,confirm_timeout,[this,enable,confirmation_generation]{const auto state=lower_control_state_.load();return control_state_generation_.load()!=confirmation_generation&&(enable?(state==3||state==4):state==2);});if(!confirmed){message=enable?"SET_ENABLE acknowledged but lower did not enter ENABLED_WAIT_COMMAND/ACTIVE":"SET_ENABLE acknowledged but lower did not finish SAFE_DAMPING and confirm DISABLED";return false;}message=enable?"all eight ports enabled":"all eight ports disabled";return true;}
  void queue_disable_no_wait(){if(!session_ready_.load()||fd_<0)return;protocol::Frame f;const auto transaction=++transaction_sequence_;if(protocol::build_set_enable(session_id_.load(),transaction,false,f)){auto pending=std::make_shared<PendingTransaction>();pending->transaction_id=transaction;pending->frame=f;{std::lock_guard<std::mutex>lock(outbound_mutex_);transactions_.push_back(pending);}wake_io();}}
  void fault_from_io(){if(control_mode_.exchange(kModeFault)!=kModeFault){standby_active_.store(false);standby_condition_.notify_all();queue_disable_no_wait();}}
  void request_disable_best_effort(){queue_disable_no_wait();std::this_thread::sleep_for(std::chrono::milliseconds(50));}

  void set_control_mode(std::uint8_t target,bool&success,std::uint8_t&actual,std::string&message){std::unique_lock<std::mutex>guard(mode_mutex_);success=false;const auto current=control_mode_.load();if(target>kModeManual){message="invalid control mode";actual=current;return;}if(target==kModeDisabled){disable_in_progress_.store(true);success=transact_enable(false,message);if(success){control_mode_.store(kModeDisabled);standby_active_.store(false);standby_condition_.notify_all();have_policy_.store(false);have_policy_source_.store(false);have_maintenance_source_.store(false);}else{control_mode_.store(kModeFault);standby_active_.store(false);standby_condition_.notify_all();have_policy_.store(false);queue_disable_no_wait();}disable_in_progress_.store(false);actual=control_mode_.load();return;}StateEnvelope state;std::string reason;if(!state_is_ready(state,reason)){message=reason;actual=current;return;}if(current==kModeDisabled&&lower_control_state_.load()!=2){message="lower controller must confirm DISABLED before enable";actual=current;return;}if(target==kModePolicy&&current!=kModeDisabled&&current!=kModeStandby){message="POLICY requires DISABLED or STANDBY";actual=current;return;}if((target==kModeSweep||target==kModeManual)&&current!=kModeDisabled){message="maintenance modes require DISABLED";actual=current;return;}if(target==kModeStandby&&current!=kModeDisabled&&current!=kModePolicy&&current!=kModeStandby){message="STANDBY requires DISABLED or POLICY";actual=current;return;}if(target==kModePolicy){have_policy_.store(false);have_policy_source_.store(false);if(current==kModeDisabled&&!transact_enable(true,message)){control_mode_.store(kModeFault);actual=kModeFault;return;}control_mode_.store(kModePolicy);success=true;message=current==kModeDisabled?"POLICY enabled without a hold/default-pose command; waiting for the first fresh policy command":"POLICY selected; waiting for a command from a new lower-state sequence";actual=target;return;}if(target==kModeSweep||target==kModeManual){capture_active_entry_positions(state);have_maintenance_source_.store(false);}
    auto hold=hold_current(state);if(current==kModeDisabled&&!transact_enable(true,message)){control_mode_.store(kModeFault);actual=kModeFault;return;}control_mode_.store(target);queue_command(target,state.state.state_sample_seq,hold,monotonic_ns());if(target==kModeStandby){standby_start_=SteadyClock::now();standby_initial_=hold;standby_active_.store(true);std::unique_lock<std::mutex>wait_lock(standby_mutex_);const auto timeout=std::chrono::duration<double>(standby_ramp_s_+2.0);if(!standby_condition_.wait_for(wait_lock,timeout,[this]{return !standby_active_.load()||control_mode_.load()!=kModeStandby;})){message="STANDBY ramp completion timeout";control_mode_.store(kModeFault);std::string ignored;transact_enable(false,ignored);actual=kModeFault;return;}if(control_mode_.load()!=kModeStandby){message="STANDBY interrupted by link or control fault";actual=control_mode_.load();return;}}success=true;message=target==kModeStandby?"full-device STANDBY ramp complete":"full-device maintenance mode enabled; lower watchdog owns command freshness";actual=target;}
  void advance_standby(const StateEnvelope&e){if(control_mode_.load()!=kModeStandby)return;auto command=hold_current(e);const auto elapsed=std::chrono::duration<double>(SteadyClock::now()-standby_start_).count();const float alpha=standby_active_.load()?std::clamp(static_cast<float>(elapsed/standby_ramp_s_),0.0F,1.0F):1.0F;for(std::size_t i=0;i<6;++i){const float goal=coordinates::lower_position_from_policy(0.0F,sign_[i],defaults_[i]);command[i].position=standby_initial_[i].position+(goal-standby_initial_[i].position)*alpha;}for(std::size_t i=6;i<8;++i)command[i].position=standby_initial_[i].position;const bool completed=standby_active_.load()&&alpha>=1;if(completed)standby_active_.store(false);queue_command(kModeStandby,e.state.state_sample_seq,command,e.receive_ns);if(completed)standby_condition_.notify_all();}
  void policy_command(const esd_link_msgs::msg::PolicyCommand&msg){const auto mode=control_mode_.load();if(mode==kModeStandby){return;}if(mode!=kModePolicy){rejected_commands_.fetch_add(1);return;}StateEnvelope state;std::string reason;if(!state_is_ready(state,reason)){rejected_commands_.fetch_add(1);return;}const auto current_gains=gains();auto command=hold_current(state);std::size_t wheel=0;for(std::size_t i=0;i<6;++i){const float value=msg.action[i];if(!std::isfinite(value)){rejected_commands_.fetch_add(1);return;}if(velocity_only_[i]){if(std::abs(value)>wheel_velocity_limits_[wheel++]){rejected_commands_.fetch_add(1);return;}command[i].position=coordinates::lower_position_from_policy(0.0F,sign_[i],defaults_[i]);command[i].velocity=coordinates::lower_direction_from_policy(value,sign_[i]);command[i].kp=0;command[i].kd=current_gains.second[i];}else{const float limited=std::clamp(value,limits_[2*i],limits_[2*i+1]);if(limited!=value){RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,"POLICY P%zu position clipped: %.6f -> %.6f",i+1,static_cast<double>(value),static_cast<double>(limited));}command[i].position=coordinates::lower_position_from_policy(limited,sign_[i],defaults_[i]);command[i].velocity=0;command[i].kp=current_gains.first[i];command[i].kd=current_gains.second[i];}}if(!accept_new_source_sequence(msg.source_state_sample_seq,last_policy_source_,have_policy_source_))return;apply_wing_target(command,state);{std::lock_guard<std::mutex>lock(policy_mutex_);latest_policy_=msg;latest_policy_ports_=command;}have_policy_.store(true);queue_command(kModePolicy,msg.source_state_sample_seq,command,msg.inference_end_monotonic_ns);}
  void apply_wing_target(std::array<ActuatorCommand,8>&command,const StateEnvelope&state){std::lock_guard<std::mutex>lock(wing_mutex_);if(!wing_initialized_){for(std::size_t i=0;i<2;++i){const auto*p=find_port(state.state,7+i);wing_target_[i]=p?coordinates::policy_direction_from_lower(p->position,wing_sign_[i]):0;}wing_initialized_=true;}for(std::size_t i=0;i<2;++i){const auto target=std::clamp(wing_target_[i],wing_limits_[2*i],wing_limits_[2*i+1]);command[6+i].position=coordinates::lower_direction_from_policy(target,wing_sign_[i]);command[6+i].velocity=0;command[6+i].kp=wing_kp_[i];command[6+i].kd=wing_kd_[i];}}
  void wing_command(const esd_link_msgs::msg::WingCommand&msg){std::lock_guard<std::mutex>lock(wing_mutex_);for(std::size_t i=0;i<2;++i){if(!std::isfinite(msg.position_rad[i])||!std::isfinite(msg.kp[i])||!std::isfinite(msg.kd[i])||msg.kp[i]<0||msg.kd[i]<0){rejected_commands_.fetch_add(1);return;}wing_target_[i]=std::clamp(msg.position_rad[i],wing_limits_[2*i],wing_limits_[2*i+1]);wing_kp_[i]=msg.kp[i];wing_kd_[i]=msg.kd[i];}wing_initialized_=true;}
  void joy_command(const sensor_msgs::msg::Joy&msg){if(msg.axes.size()<=4)return;const float raw=std::abs(msg.axes[4])>0.1F?msg.axes[4]:0.0F;wing_velocity_.store(raw*0.7853981634F);last_joy_ns_.store(monotonic_ns());}
  void advance_wing(const StateEnvelope&e){if(!wing_rc_enabled_.load()||control_mode_.load()!=kModePolicy||!have_policy_.load())return;const auto now_ns=monotonic_ns();if(timestamp_is_stale(now_ns,last_joy_ns_.load(),500000000ULL))return;const auto velocity=wing_velocity_.load();if(std::abs(velocity)<1e-5F)return;std::lock_guard<std::mutex>lock(wing_mutex_);if(!wing_initialized_){const auto*p7=find_port(e.state,7);const auto*p8=find_port(e.state,8);wing_target_={p7?coordinates::policy_direction_from_lower(p7->position,wing_sign_[0]):0,p8?coordinates::policy_direction_from_lower(p8->position,wing_sign_[1]):0};wing_initialized_=true;}wing_target_[0]=std::clamp(wing_target_[0]+velocity*0.004F,wing_limits_[0],wing_limits_[1]);wing_target_[1]=std::clamp(wing_target_[1]-velocity*0.004F,wing_limits_[2],wing_limits_[3]);}
  void maintenance_command(const esd_link_msgs::msg::MaintenanceCommand&msg){const auto mode=control_mode_.load();if((mode!=kModeSweep&&mode!=kModeManual)||msg.mode!=mode){rejected_commands_.fetch_add(1);return;}StateEnvelope state;std::string reason;if(!state_is_ready(state,reason)){rejected_commands_.fetch_add(1);return;}std::array<ActuatorCommand,8> command{};std::uint32_t seen=0;for(std::size_t i=0;i<8;++i){const auto id=msg.port_id[i];if(id<1||id>8||(seen&(1U<<id))){rejected_commands_.fetch_add(1);return;}seen|=1U<<id;if(id<=6){const auto index=static_cast<std::size_t>(id-1);if((velocity_only_[index]&&std::abs(msg.velocity_rad_s[i])>wheel_velocity_limits_[index==2?0:1])||(!velocity_only_[index]&&!maintenance_position_target_is_safe(id,msg.position_rad[i],limits_[2*index],limits_[2*index+1]))){rejected_commands_.fetch_add(1);return;}command[i]={id,coordinates::lower_position_from_policy(msg.position_rad[i],sign_[index],defaults_[index]),coordinates::lower_direction_from_policy(msg.velocity_rad_s[i],sign_[index]),msg.kp[i],msg.kd[i],coordinates::lower_direction_from_policy(msg.effort_nm[i],sign_[index])};}else{const auto wing=static_cast<std::size_t>(id-7);if(!maintenance_position_target_is_safe(id,msg.position_rad[i],wing_limits_[2*wing],wing_limits_[2*wing+1])){rejected_commands_.fetch_add(1);return;}command[i]={id,coordinates::lower_direction_from_policy(msg.position_rad[i],wing_sign_[wing]),coordinates::lower_direction_from_policy(msg.velocity_rad_s[i],wing_sign_[wing]),msg.kp[i],msg.kd[i],coordinates::lower_direction_from_policy(msg.effort_nm[i],wing_sign_[wing])};}}if(seen!=expected_mask_){rejected_commands_.fetch_add(1);return;}if(!accept_new_source_sequence(msg.source_state_sample_seq,last_maintenance_source_,have_maintenance_source_))return;queue_command(mode,msg.source_state_sample_seq,command,monotonic_ns());}

  void native_set_zeros(const esd_link_msgs::srv::SetLowerZeros::Request&req,esd_link_msgs::srv::SetLowerZeros::Response&res){std::lock_guard<std::mutex>guard(mode_mutex_);if(control_mode_.load()!=kModeDisabled||lower_control_state_.load()!=2){res.success=false;res.message="SET_ZERO requires upper and lower DISABLED";return;}if(req.port_ids.empty()||req.port_ids.size()!=req.assigned_position_rad.size()||req.port_ids.size()>8){res.success=false;res.message="invalid zero assignment lengths";return;}protocol::Frame frame;const auto tx=++transaction_sequence_;if(!protocol::build_set_zero(session_id_.load(),tx,req.persist,req.port_ids.data(),req.assigned_position_rad.data(),req.port_ids.size(),frame)){res.success=false;res.message="invalid zero assignments";return;}auto pending=submit_transaction(frame,tx,std::chrono::seconds(10));if(!pending->done){res.success=false;res.message="SET_ZERO timeout";return;}res.config_fingerprint=pending->result.config_fingerprint;res.success=pending->result.status==0;res.message=res.success?"zeros stored; identity recorded and session will be reopened":std::string("SET_ZERO rejected: ")+protocol::status_name(pending->result.status);if(res.success){std::string persist_error;if(!persist_identity_record(res.config_fingerprint,persist_error))res.message="zeros stored, but host identity record failed: "+persist_error;expected_fingerprint_.store(res.config_fingerprint);session_ready_.store(false);reopen_session_.store(true);wake_io();}}
  void set_gains(const motors::srv::SetMotorGains::Request&req,motors::srv::SetMotorGains::Response&res){std::lock_guard<std::mutex>lock(gain_mutex_);if(req.restore_defaults){kp_=default_kp_;kd_=default_kd_;res.success=true;res.message="restored ESD-Link startup gains";return;}if(req.kp.size()!=req.kd.size()||req.kp.empty()){res.success=false;res.message="kp/kd size mismatch";return;}std::vector<int>ids;if(req.motor_ids.empty()){if(req.kp.size()!=6){res.success=false;res.message="empty motor_ids requires six gains";return;}ids={1,2,3,4,5,6};}else{for(auto id:req.motor_ids)ids.push_back(id);}if(ids.size()!=req.kp.size()){res.success=false;res.message="motor_ids/kp/kd size mismatch";return;}for(std::size_t i=0;i<ids.size();++i){if(ids[i]<1||ids[i]>6||!std::isfinite(req.kp[i])||!std::isfinite(req.kd[i])||req.kp[i]<0||req.kd[i]<0){res.success=false;res.message="invalid gain assignment";return;}kp_[ids[i]-1]=req.kp[i];kd_[ids[i]-1]=req.kd[i];}res.success=true;res.message="ESD-Link runtime gains updated";}
  void read_motors(motors::srv::ReadMotors::Response&res){StateEnvelope state;std::string reason;if(!state_is_ready(state,reason)){res.success=false;res.message=reason;return;}std::ostringstream out;for(std::uint8_t id=1;id<=8;++id){const auto*p=find_port(state.state,id);if(id<=6){const auto index=static_cast<std::size_t>(id-1);auto position=coordinates::policy_position_from_lower(p->position,sign_[index],defaults_[index]);if(velocity_only_[index])position=coordinates::wrap_to_limits(position,limits_[2*index],limits_[2*index+1]);out<<"P"<<static_cast<int>(id)<<" q="<<position<<" dq="<<coordinates::policy_direction_from_lower(p->velocity,sign_[index])<<" tau="<<coordinates::policy_direction_from_lower(p->effort,sign_[index]);}else{const auto wing=static_cast<std::size_t>(id-7);out<<"P"<<static_cast<int>(id)<<" q="<<coordinates::policy_direction_from_lower(p->position,wing_sign_[wing])<<" dq="<<coordinates::policy_direction_from_lower(p->velocity,wing_sign_[wing])<<" tau="<<coordinates::policy_direction_from_lower(p->effort,wing_sign_[wing]);}out<<(id==8?"":"; ");}res.success=true;res.message=out.str();}
  void manual_control(const motors::srv::ControlMotor::Request&req,motors::srv::ControlMotor::Response&res){if(control_mode_.load()!=kModeManual){res.success=false;res.message="control_motor requires MANUAL_TEST mode";return;}if(req.motor_id<1||req.motor_id>8){res.success=false;res.message="motor_id must be 1..8";return;}StateEnvelope state;std::string reason;if(!state_is_ready(state,reason)){res.success=false;res.message=reason;return;}auto command=hold_current(state);auto&port=command[req.motor_id-1];if(req.motor_id<=6){const auto index=static_cast<std::size_t>(req.motor_id-1);if((velocity_only_[index]&&std::abs(req.velocity)>wheel_velocity_limits_[index==2?0:1])||(!velocity_only_[index]&&(req.position<limits_[2*index]||req.position>limits_[2*index+1]))){res.success=false;res.message="manual target outside configured limit";return;}port.position=coordinates::lower_position_from_policy(req.position,sign_[index],defaults_[index]);port.velocity=coordinates::lower_direction_from_policy(req.velocity,sign_[index]);port.effort=coordinates::lower_direction_from_policy(req.effort,sign_[index]);}else{const auto wing=static_cast<std::size_t>(req.motor_id-7);if(req.position<wing_limits_[2*wing]||req.position>wing_limits_[2*wing+1]){res.success=false;res.message="manual wing target outside configured limit";return;}port.position=coordinates::lower_direction_from_policy(req.position,wing_sign_[wing]);port.velocity=coordinates::lower_direction_from_policy(req.velocity,wing_sign_[wing]);port.effort=coordinates::lower_direction_from_policy(req.effort,wing_sign_[wing]);}if(!accept_new_source_sequence(state.state.state_sample_seq,last_maintenance_source_,have_maintenance_source_)||!queue_command(kModeManual,state.state.state_sample_seq,command,monotonic_ns())){res.success=false;res.message="manual command invalid or does not use a new lower-state sequence";return;}res.success=true;res.message="full P1-P8 command queued once; only selected target changed";}

  void publish_link_status(){esd_link_msgs::msg::LinkStatus msg;msg.header.stamp=now();msg.control_mode=control_mode_.load();msg.link_state=!connected_.load()?msg.DISCONNECTED:(!session_ready_.load()?msg.CONNECTING:(msg.control_mode==kModeDisabled?msg.READ_ONLY:(msg.control_mode==kModeFault?msg.FAULT:msg.READY)));msg.device_control_state=device_control_state_.load();msg.session_valid=session_ready_.load();msg.session_id=session_id_.load();msg.schema_id=schema_id_;msg.config_fingerprint=fingerprint_.load();msg.active_port_mask=expected_mask_;msg.offline_port_mask=offline_mask_.load();msg.fault_flags=fault_flags_.load();msg.last_reject_code=last_reject_code_.load();const auto now_ns=monotonic_ns(),last=last_state_ns_.load();msg.latest_state_age_ms=last?static_cast<float>(elapsed_ns_saturated(now_ns,last))/1e6F:std::numeric_limits<float>::infinity();const auto count=state_count_.load();const auto previous=last_status_state_count_.exchange(count);msg.state_rate_hz=static_cast<float>(count-previous);msg.maximum_interarrival_ms=static_cast<float>(max_gap_ns_.exchange(0))/1e6F;msg.valid_frames=decoder_.frames();msg.cobs_errors=decoder_.cobs_errors();msg.crc_errors=decoder_.crc_errors();msg.sequence_gaps=sequence_gaps_.load();msg.rejected_commands=rejected_commands_.load()+device_rejected_commands_.load();msg.watchdog_events=watchdog_events_.load();msg.control_fault_events=control_fault_events_.load();msg.reconnects=reconnects_.load();msg.device_valid_commands=device_valid_commands_.load();msg.device_invalid_frames=device_invalid_frames_.load();msg.device_rejected_commands=device_rejected_commands_.load();msg.device_command_age_ms=device_command_age_ms_.load();msg.last_command_sequence=last_sent_command_.load();msg.last_applied_command_sequence=last_applied_sequence_.load();msg.command_to_applied_ms=static_cast<float>(last_apply_latency_ns_.load())/1e6F;msg.observation_to_command_ms=static_cast<float>(last_observation_to_command_ns_.load())/1e6F;if(!msg.session_valid)msg.message="ESD-Link session unavailable";else if(msg.fault_flags||msg.offline_port_mask){std::ostringstream reason;reason<<"ESD-Link lower fault=0x"<<std::hex<<msg.fault_flags<<" offline=0x"<<msg.offline_port_mask;msg.message=reason.str();}else msg.message="ESD-Link session ready; timing fields are diagnostic only";link_pub_->publish(msg);}

  protocol::StreamDecoder decoder_;std::string serial_device_,identity_record_path_;std::uint16_t layout_id_{1},schema_id_{1},expected_rate_{250};std::uint32_t expected_mask_{0x1FE};std::atomic<std::uint32_t>expected_fingerprint_{0};int rt_priority_{70},cpu_affinity_{-1},enable_confirm_timeout_ms_{1000},disable_confirm_timeout_ms_{6500};double standby_ramp_s_{3};
  std::array<float,6>defaults_{},sign_{},kp_{},kd_{},default_kp_{},default_kd_{};std::array<bool,6>velocity_only_{};std::array<float,12>limits_{};std::array<float,2>wheel_velocity_limits_{},wing_sign_{};std::array<float,4>wing_limits_{};std::array<float,2>wing_kp_{},wing_kd_{},wing_target_{};std::array<float,8>active_entry_positions_{};
  int fd_{-1},wake_fd_{-1};std::atomic<bool>running_{false},connected_{false},session_ready_{false},reopen_session_{false},disable_in_progress_{false};std::thread io_thread_,publish_thread_;std::uint32_t host_nonce_{0};std::atomic<std::uint32_t>session_id_{0},fingerprint_{0},lower_boot_id_{0},tx_sequence_{0},transaction_sequence_{0};
  LatestMailbox<StateEnvelope>state_mailbox_;std::mutex publish_mutex_;std::condition_variable publish_condition_;std::mutex cache_mutex_,control_state_mutex_,standby_mutex_;std::condition_variable control_state_condition_,standby_condition_;StateEnvelope latest_state_{};bool have_state_{false};
  std::mutex outbound_mutex_,pending_mutex_;std::deque<std::shared_ptr<PendingTransaction>>transactions_;std::vector<std::shared_ptr<PendingTransaction>>pending_;std::optional<QueuedCommand>queued_command_;
  std::atomic<std::uint8_t>control_mode_{kModeDisabled};std::mutex mode_mutex_,gain_mutex_,policy_mutex_,wing_mutex_;std::atomic<bool>standby_active_{false},wing_rc_enabled_{false},have_policy_{false},have_policy_source_{false},have_maintenance_source_{false};SteadyClock::time_point standby_start_;std::array<ActuatorCommand,8>standby_initial_{},latest_policy_ports_{};esd_link_msgs::msg::PolicyCommand latest_policy_{};bool wing_initialized_{false};std::atomic<float>wing_velocity_{0};std::atomic<std::uint64_t>last_joy_ns_{0};std::atomic<std::uint32_t>last_policy_source_{0},last_maintenance_source_{0};
  std::atomic<std::uint64_t>state_count_{0},last_status_state_count_{0},last_state_ns_{0},max_gap_ns_{0},sequence_gaps_{0},rejected_commands_{0},reconnects_{0},watchdog_events_{0},control_fault_events_{0},last_apply_latency_ns_{0},last_observation_to_command_ns_{0},control_state_generation_{0};std::atomic<std::uint32_t>last_state_sequence_{0},last_sent_command_{0},last_applied_sequence_{0},device_valid_commands_{0},device_invalid_frames_{0},device_rejected_commands_{0},fault_flags_{0},offline_mask_{0};std::atomic<std::uint16_t>last_reject_code_{0},device_command_age_ms_{0};std::atomic<std::uint8_t>device_control_state_{0},lower_control_state_{0};std::array<AckEntry,256>acks_{};std::array<StateTime,256>state_times_{};std::mutex ack_mutex_;
  rclcpp::Publisher<esd_link_msgs::msg::LowerState>::SharedPtr lower_pub_;rclcpp::Publisher<esd_link_msgs::msg::LinkStatus>::SharedPtr link_pub_;rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joints_pub_,wings_pub_;rclcpp::Publisher<motors::msg::MotorRuntimeStatus>::SharedPtr motor_status_pub_;rclcpp::Publisher<motors::msg::WingRuntimeStatus>::SharedPtr wing_status_pub_;
  rclcpp::Subscription<esd_link_msgs::msg::PolicyCommand>::SharedPtr policy_sub_;rclcpp::Subscription<esd_link_msgs::msg::MaintenanceCommand>::SharedPtr maintenance_sub_;rclcpp::Subscription<esd_link_msgs::msg::WingCommand>::SharedPtr wing_sub_;rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Service<esd_link_msgs::srv::SetControlMode>::SharedPtr control_mode_service_;rclcpp::Service<esd_link_msgs::srv::SetLowerZeros>::SharedPtr zero_native_service_;rclcpp::Service<motors::srv::SetOperationalState>::SharedPtr operational_service_;rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr soft_disarm_service_;rclcpp::Service<motors::srv::SetZeros>::SharedPtr zero_compat_service_;rclcpp::Service<motors::srv::SetMotorGains>::SharedPtr gains_service_;rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr wing_rc_service_;rclcpp::Service<motors::srv::ReadMotors>::SharedPtr read_service_;rclcpp::Service<motors::srv::ControlMotor>::SharedPtr control_service_;rclcpp::Service<motors::srv::ResetMotors>::SharedPtr unsupported_reset_;rclcpp::Service<motors::srv::ClearErrors>::SharedPtr unsupported_clear_;rclcpp::Service<motors::srv::IdentifyMotorID>::SharedPtr unsupported_identify_;rclcpp::Service<motors::srv::SetMasterID>::SharedPtr unsupported_master_id_;rclcpp::Service<motors::srv::ScanMotors>::SharedPtr unsupported_scan_;rclcpp::Service<motors::srv::RunSafetyCheck>::SharedPtr unsupported_safety_;rclcpp::TimerBase::SharedPtr diagnostics_timer_;
};

}  // namespace esd_link_bridge

int main(int argc,char**argv){rclcpp::init(argc,argv);try{auto node=std::make_shared<esd_link_bridge::EsdLinkBridgeNode>();rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),4);executor.add_node(node);executor.spin();}catch(const std::exception&e){std::fprintf(stderr,"esd_link_bridge fatal: %s\n",e.what());rclcpp::shutdown();return 1;}rclcpp::shutdown();return 0;}
