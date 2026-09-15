cmake_minimum_required(VERSION 3.8)
project(esd_link_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LinkStatus.msg"
  "msg/LowerState.msg"
  "msg/MaintenanceCommand.msg"
  "msg/PolicyCommand.msg"
  "msg/WingCommand.msg"
  "srv/SetControlMode.srv"
  "srv/SetLowerZeros.srv"
  DEPENDENCIES builtin_interfaces sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
