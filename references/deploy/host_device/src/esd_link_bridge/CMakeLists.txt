cmake_minimum_required(VERSION 3.8)
project(esd_link_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(esd_link_msgs REQUIRED)
find_package(motors REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_library(esd_link_protocol src/protocol.cpp)
target_include_directories(esd_link_protocol PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

add_executable(esd_link_bridge_node src/esd_link_bridge_node.cpp)
target_link_libraries(esd_link_bridge_node esd_link_protocol pthread)
target_include_directories(esd_link_bridge_node PRIVATE include)
ament_target_dependencies(esd_link_bridge_node
  esd_link_msgs motors rclcpp sensor_msgs std_msgs std_srvs)

add_executable(esd_link_emergency_disable src/emergency_disable.cpp)
target_link_libraries(esd_link_emergency_disable esd_link_protocol pthread)
target_include_directories(esd_link_emergency_disable PRIVATE include)

install(TARGETS esd_link_protocol esd_link_bridge_node esd_link_emergency_disable
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_gtest(test_protocol test/test_protocol.cpp)
  target_link_libraries(test_protocol esd_link_protocol)
  target_include_directories(test_protocol PRIVATE include)
  ament_add_gtest(test_latest_mailbox test/test_latest_mailbox.cpp)
  target_include_directories(test_latest_mailbox PRIVATE include)
  target_link_libraries(test_latest_mailbox pthread)
  ament_add_gtest(test_coordinates test/test_coordinates.cpp)
  target_include_directories(test_coordinates PRIVATE include)
  ament_add_gtest(test_time_utils test/test_time_utils.cpp)
  target_include_directories(test_time_utils PRIVATE include)
  ament_add_pytest_test(test_bridge_pty test/test_bridge_pty.py TIMEOUT 30)
endif()

ament_export_include_directories(include)
ament_export_libraries(esd_link_protocol)
ament_package()
