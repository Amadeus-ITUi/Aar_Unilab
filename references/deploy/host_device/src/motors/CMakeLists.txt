cmake_minimum_required(VERSION 3.8)
project(motors)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotorRuntimeStatus.msg"
  "msg/WingRuntimeStatus.msg"
  "srv/ControlMotor.srv"
  "srv/ResetMotors.srv"
  "srv/ReadMotors.srv"
  "srv/SetZeros.srv"
  "srv/ClearErrors.srv"
  "srv/IdentifyMotorID.srv"
  "srv/ScanMotors.srv"
  "srv/SetMasterID.srv"
  "srv/SetMotorGains.srv"
  "srv/RunSafetyCheck.srv"
  "srv/SetOperationalState.srv"
  DEPENDENCIES builtin_interfaces
)

add_library(motors_lib SHARED 
  src/motor_driver.cpp
  src/rs00_motor_driver.cpp
  src/rs05_motor_driver.cpp
  src/el05_motor_driver.cpp
  src/SocketCAN.cpp
)
set(PUBLIC_DEPENDENCIES
    fmt::fmt spdlog::spdlog pthread Eigen3::Eigen)
target_include_directories(motors_lib
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(motors_lib PUBLIC ${PUBLIC_DEPENDENCIES})

add_executable(${PROJECT_NAME}_node 
                src/motors_node.cpp)
target_include_directories(${PROJECT_NAME}_node
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}_node PUBLIC ${PUBLIC_DEPENDENCIES} motors_lib)
ament_target_dependencies(${PROJECT_NAME}_node PUBLIC rclcpp sensor_msgs std_msgs std_srvs rosidl_default_runtime)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_node PUBLIC "${cpp_typesupport_target}")

set_target_properties(${PROJECT_NAME}_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

add_executable(wing_motor_node src/wing_motor_node.cpp)
target_include_directories(wing_motor_node
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(wing_motor_node PUBLIC ${PUBLIC_DEPENDENCIES} motors_lib)
target_link_libraries(wing_motor_node PUBLIC "${cpp_typesupport_target}")
ament_target_dependencies(wing_motor_node PUBLIC rclcpp rcl_interfaces sensor_msgs std_srvs)
set_target_properties(wing_motor_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_wing_control_logic test/test_wing_control_logic.cpp)
  target_include_directories(test_wing_control_logic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  ament_add_gtest(test_soft_disarm_logic test/test_soft_disarm_logic.cpp)
  target_include_directories(test_soft_disarm_logic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()

install(TARGETS motors_lib
  LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(TARGETS wing_motor_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})


install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)


ament_export_dependencies(rclcpp rcl_interfaces sensor_msgs std_msgs std_srvs builtin_interfaces rosidl_default_runtime rosidl_default_generators)
ament_export_libraries(motors_lib)
ament_package()
