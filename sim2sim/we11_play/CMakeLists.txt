cmake_minimum_required(VERSION 3.16)
project(walking_eagle_we11_play VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

set(BUNDLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(RL_SOURCE_DIR "${BUNDLE_ROOT}/src/rl_sar")
set(AAR_WE11_RUNTIME_ROOT "" CACHE PATH "WE11 Play runtime root outside the Git repository")
set(AAR_MUJOCO_ROOT "${AAR_WE11_RUNTIME_ROOT}/mujoco")
set(AAR_ONNXRUNTIME_ROOT "${AAR_WE11_RUNTIME_ROOT}/inference_runtime/onnxruntime")
set(AAR_GLFW_ROOT "${AAR_WE11_RUNTIME_ROOT}/glfw")

find_package(Threads REQUIRED)
find_package(TBB REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)
find_package(yaml-cpp REQUIRED)

find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h PATHS "${AAR_MUJOCO_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(MUJOCO_LIBRARY NAMES libmujoco.so.3.2.7 mujoco PATHS "${AAR_MUJOCO_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(ONNXRUNTIME_LIBRARY NAMES libonnxruntime.so.1.22.0 onnxruntime PATHS "${AAR_ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(GLFW_LIBRARY NAMES libglfw.so.3.4 glfw PATHS "${AAR_GLFW_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(onnxruntime_prebuilt SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime_prebuilt PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${AAR_ONNXRUNTIME_ROOT}/include"
)

add_library(mujoco_prebuilt SHARED IMPORTED GLOBAL)
set_target_properties(mujoco_prebuilt PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
)

add_library(glfw_prebuilt SHARED IMPORTED GLOBAL)
set_target_properties(glfw_prebuilt PROPERTIES
    IMPORTED_LOCATION "${GLFW_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${AAR_GLFW_ROOT}/include"
)

set(RL_CORE "${RL_SOURCE_DIR}/library/core")
set(RL_INCLUDE_DIRS
    "${RL_SOURCE_DIR}/include"
    "${RL_CORE}/matplotlibcpp"
    "${RL_CORE}/observation_buffer"
    "${RL_CORE}/rl_sdk"
    "${RL_CORE}/loop"
    "${RL_CORE}/fsm"
    "${RL_CORE}/vector_math"
    "${RL_CORE}/motion_loader"
    "${RL_CORE}/inference_runtime"
    "${RL_CORE}/logger"
    "${RL_SOURCE_DIR}/fsm_robot"
    "${RL_SOURCE_DIR}/library/thirdparty/mujoco_simulate"
    "${RL_SOURCE_DIR}/library/thirdparty/joystick"
)

add_library(inference_runtime
    "${RL_CORE}/inference_runtime/inference_runtime.cpp"
)
target_include_directories(inference_runtime PUBLIC ${RL_INCLUDE_DIRS})
target_compile_definitions(inference_runtime PUBLIC USE_ONNX)
target_link_libraries(inference_runtime PUBLIC onnxruntime_prebuilt)

add_library(motion_loader
    "${RL_CORE}/motion_loader/motion_loader.cpp"
)
target_include_directories(motion_loader PUBLIC ${RL_INCLUDE_DIRS})

add_library(rl_sdk
    "${RL_CORE}/rl_sdk/rl_sdk.cpp"
)
target_include_directories(rl_sdk PUBLIC ${RL_INCLUDE_DIRS})
target_compile_definitions(rl_sdk PUBLIC
    USE_CMAKE
    USE_ONNX
    POLICY_DIR="${BUNDLE_ROOT}/policy"
)
target_link_libraries(rl_sdk PUBLIC
    motion_loader
    inference_runtime
    Python3::Python
    Python3::Module
    TBB::tbb
)
if(Python3_NumPy_FOUND)
    target_link_libraries(rl_sdk PUBLIC Python3::NumPy)
else()
    target_compile_definitions(rl_sdk PUBLIC WITHOUT_NUMPY)
endif()

add_library(observation_buffer
    "${RL_CORE}/observation_buffer/observation_buffer.cpp"
)
target_include_directories(observation_buffer PUBLIC ${RL_INCLUDE_DIRS})

file(GLOB MUJOCO_SIMULATE_SOURCES
    "${RL_SOURCE_DIR}/library/thirdparty/mujoco_simulate/*.cc"
)

add_executable(rl_sim_mujoco
    "${RL_SOURCE_DIR}/src/rl_sim_mujoco.cpp"
    ${MUJOCO_SIMULATE_SOURCES}
    "${RL_SOURCE_DIR}/library/thirdparty/joystick/joystick.cc"
)
target_include_directories(rl_sim_mujoco PRIVATE
    ${RL_INCLUDE_DIRS}
    "${MUJOCO_INCLUDE_DIR}"
)
target_compile_definitions(rl_sim_mujoco PRIVATE
    USE_CMAKE
    USE_MUJOCO
    USE_ONNX
    BOOST_BIND_GLOBAL_PLACEHOLDERS
    CMAKE_CURRENT_SOURCE_DIR="${RL_SOURCE_DIR}"
    POLICY_DIR="${BUNDLE_ROOT}/policy"
)
target_link_libraries(rl_sim_mujoco PRIVATE
    rl_sdk
    observation_buffer
    yaml-cpp
    Threads::Threads
    mujoco_prebuilt
    glfw_prebuilt
)
set_target_properties(rl_sim_mujoco PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH
        "${AAR_ONNXRUNTIME_ROOT}/lib;${AAR_MUJOCO_ROOT}/lib;${AAR_GLFW_ROOT}/lib"
)
