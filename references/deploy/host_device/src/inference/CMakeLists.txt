cmake_minimum_required(VERSION 3.8)
project(inference)

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

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Boost COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(esd_link_msgs REQUIRED)

# MNN 库配置
# 检查 MNN 库文件是否存在
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(MNN_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mnn-linux-x64)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(MNN_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mnn-linux-aarch64)
else()
    set(MNN_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mnn)
endif()

set(MNN_INCLUDE_DIR ${MNN_ROOT_DIR}/include)
set(MNN_LIB_DIR ${MNN_ROOT_DIR}/lib)

# 检查 MNN 库文件是否存在
set(MNN_LIB_FILE ${MNN_LIB_DIR}/libMNN.so)

# 尝试从系统路径查找
if(NOT EXISTS ${MNN_LIB_FILE})
    find_library(MNN_LIB MNN 
        PATHS 
        /usr/local/lib
        /usr/lib
        ${MNN_LIB_DIR}
    )
    
    find_path(MNN_INCLUDE_DIRS 
        NAMES MNN/Interpreter.hpp
        PATHS
        /usr/local/include
        /usr/include
        ${MNN_INCLUDE_DIR}
    )
    
    if(MNN_LIB AND MNN_INCLUDE_DIRS)
        set(MNN_LIB_FILE ${MNN_LIB})
        set(BUILD_INFERENCE_NODE ON)
        message(STATUS "找到 MNN 库: ${MNN_LIB}")
        message(STATUS "找到 MNN 头文件: ${MNN_INCLUDE_DIRS}")
    else()
        set(BUILD_INFERENCE_NODE OFF)
        message(WARNING "MNN 库文件不存在: ${MNN_LIB_FILE}")
        message(WARNING "lab_inference_node 将不会被编译。请安装 MNN 库或将其放置到 thirdparty/mnn 目录。")
    endif()
else()
    # 使用 thirdparty 目录中的 MNN
    set(MNN_LIB ${MNN_LIB_FILE})
    set(MNN_INCLUDE_DIRS ${MNN_INCLUDE_DIR})
    set(BUILD_INFERENCE_NODE ON)
    message(STATUS "找到 MNN 库: ${MNN_LIB_FILE}")
    message(STATUS "找到 MNN 头文件: ${MNN_INCLUDE_DIR}")
endif()

set(PUBLIC_DEPENDENCIES
    ${Boost_LIBRARIES} pthread Eigen3::Eigen)

# 只有在库文件存在时才编译 Lab 推理节点
if(BUILD_INFERENCE_NODE)
add_executable(lab_inference_node src/lab_inference_node.cpp)
target_include_directories(lab_inference_node
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  ${MNN_INCLUDE_DIRS}
)
if(EXISTS ${MNN_LIB_FILE})
    target_link_libraries(lab_inference_node PUBLIC ${PUBLIC_DEPENDENCIES} ${MNN_LIB_FILE})
else()
    target_link_libraries(lab_inference_node PUBLIC ${PUBLIC_DEPENDENCIES} ${MNN_LIB})
endif()
ament_target_dependencies(lab_inference_node PUBLIC rclcpp sensor_msgs geometry_msgs std_msgs std_srvs ament_index_cpp esd_link_msgs)
set_target_properties(lab_inference_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)
install(TARGETS lab_inference_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})
else()
    message(STATUS "跳过 lab_inference_node 的编译（缺少 MNN 库）")
endif()

# 扫频曲线下发节点（用于电机测试）
add_executable(sweep_frequency_node src/sweep_frequency_node.cpp)
target_include_directories(sweep_frequency_node
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(sweep_frequency_node PUBLIC rclcpp sensor_msgs std_msgs)
set_target_properties(sweep_frequency_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" 
)

install(TARGETS sweep_frequency_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY scripts
  DESTINATION share/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

if(BUILD_TESTING)
    add_executable(lab_deploy_utils_test test/lab_deploy_utils_test.cpp)
    target_include_directories(lab_deploy_utils_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME lab_deploy_utils_test COMMAND lab_deploy_utils_test)
endif()
# 只有在库文件存在时才安装库文件
if(BUILD_INFERENCE_NODE AND EXISTS ${MNN_LIB_FILE})
    install(FILES ${MNN_LIB_FILE} 
            DESTINATION lib
            OPTIONAL)
    # 尝试安装符号链接（如果存在）
    if(EXISTS ${MNN_LIB_DIR}/libMNN.so.1)
        install(FILES ${MNN_LIB_DIR}/libMNN.so.1
                DESTINATION lib
                OPTIONAL)
    endif()
endif()

ament_export_dependencies(rclcpp sensor_msgs geometry_msgs std_msgs std_srvs)
ament_export_libraries(${PROJECT_NAME})
ament_package()
