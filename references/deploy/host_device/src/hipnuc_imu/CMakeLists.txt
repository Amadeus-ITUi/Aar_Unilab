cmake_minimum_required(VERSION 3.5)
project(hipnuc_imu)

# 统一使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_index_cpp REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ========== imu_node ==========
add_executable(imu_node src/serial_port.cpp)
ament_target_dependencies(imu_node rclcpp std_msgs sensor_msgs ament_index_cpp)

# 关键修复：链接 stdc++fs 和 Eigen3
target_link_libraries(imu_node 
    stdc++fs              # 添加这一行！关键！
    Eigen3::Eigen
)

target_include_directories(imu_node PRIVATE ${EIGEN3_INCLUDE_DIRS})

set_target_properties(imu_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

# ========== imu_node_debug ==========
add_executable(imu_node_debug src/serial_port_debug.cpp)
ament_target_dependencies(imu_node_debug rclcpp std_msgs sensor_msgs ament_index_cpp)

target_link_libraries(imu_node_debug 
    stdc++fs              # 同样添加
    Eigen3::Eigen
)
target_include_directories(imu_node_debug PRIVATE ${EIGEN3_INCLUDE_DIRS})

set_target_properties(imu_node_debug PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

# ========== listener ==========
add_executable(listener src/sub_spec.cpp)
ament_target_dependencies(listener rclcpp std_msgs sensor_msgs)

# ========== Install ==========
install(TARGETS imu_node imu_node_debug listener  
    DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY
  config
  scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
