cmake_minimum_required(VERSION 3.16)
project(aar_unilab_sim2sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(aar_runtime STATIC
  src/deployment_contract.cpp
  src/npy.cpp
  src/telemetry.cpp
)
target_include_directories(aar_runtime PUBLIC include)
target_compile_options(aar_runtime PRIVATE -Wall -Wextra -Wpedantic)

add_executable(aar_contract_check src/main.cpp)
target_link_libraries(aar_contract_check PRIVATE aar_runtime)

set(AAR_MUJOCO_ROOT "" CACHE PATH "MuJoCo Python package root containing include/ and libmujoco")
set(AAR_ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime Python package capi directory")
set(AAR_ONNXRUNTIME_INCLUDE "" CACHE PATH "ONNX Runtime C/C++ API include directory")
set(AAR_GLFW_ROOT "" CACHE PATH "Python glfw package root containing x11/libglfw.so")
set(AAR_GLFW_INCLUDE "" CACHE PATH "GLFW include directory")
if(AAR_MUJOCO_ROOT)
  find_path(AAR_MUJOCO_INCLUDE mujoco/mujoco.h PATHS ${AAR_MUJOCO_ROOT}/include NO_DEFAULT_PATH REQUIRED)
  find_library(AAR_MUJOCO_LIBRARY NAMES libmujoco.so.3.8.0 mujoco PATHS ${AAR_MUJOCO_ROOT} NO_DEFAULT_PATH REQUIRED)
  add_executable(aar_mujoco_smoke src/mujoco_smoke.cpp)
  target_include_directories(aar_mujoco_smoke PRIVATE ${AAR_MUJOCO_INCLUDE})
  target_link_libraries(aar_mujoco_smoke PRIVATE ${AAR_MUJOCO_LIBRARY})
  set_target_properties(aar_mujoco_smoke PROPERTIES BUILD_RPATH ${AAR_MUJOCO_ROOT})
endif()

if(AAR_MUJOCO_ROOT AND AAR_ONNXRUNTIME_ROOT AND AAR_ONNXRUNTIME_INCLUDE)
  find_library(AAR_ONNXRUNTIME_LIBRARY
    NAMES onnxruntime libonnxruntime.so.1.22.0
    PATHS ${AAR_ONNXRUNTIME_ROOT}
    NO_DEFAULT_PATH REQUIRED
  )
  add_executable(aar_sim2sim src/sim2sim_main.cpp src/onnx_actor.cpp)
  target_include_directories(aar_sim2sim PRIVATE
    ${AAR_MUJOCO_INCLUDE}
    ${AAR_ONNXRUNTIME_INCLUDE}
  )
  target_link_libraries(aar_sim2sim PRIVATE aar_runtime ${AAR_MUJOCO_LIBRARY} ${AAR_ONNXRUNTIME_LIBRARY})
  set_target_properties(aar_sim2sim PROPERTIES
    BUILD_RPATH "${AAR_MUJOCO_ROOT};${AAR_ONNXRUNTIME_ROOT}"
  )
  if(AAR_GLFW_ROOT AND AAR_GLFW_INCLUDE)
    find_library(AAR_GLFW_LIBRARY NAMES glfw libglfw.so
      PATHS ${AAR_GLFW_ROOT}/x11 ${AAR_GLFW_ROOT}/wayland NO_DEFAULT_PATH REQUIRED)
    target_include_directories(aar_sim2sim PRIVATE ${AAR_GLFW_INCLUDE})
    target_compile_definitions(aar_sim2sim PRIVATE AAR_WITH_GLFW=1)
    target_link_libraries(aar_sim2sim PRIVATE ${AAR_GLFW_LIBRARY})
    set_property(TARGET aar_sim2sim APPEND PROPERTY BUILD_RPATH
      "${AAR_GLFW_ROOT}/x11;${AAR_GLFW_ROOT}/wayland")
  endif()
endif()

enable_testing()
add_test(
  NAME contract_accepts_v1
  COMMAND aar_contract_check
    ${CMAKE_CURRENT_SOURCE_DIR}/../releases/examples/we11_flat/deployment_manifest.json
)
set_tests_properties(contract_accepts_v1 PROPERTIES PASS_REGULAR_EXPRESSION "aar-unilab.actor.v1")
add_test(
  NAME contract_rejects_unknown
  COMMAND aar_contract_check ${CMAKE_CURRENT_SOURCE_DIR}/tests/unknown_manifest.json
)
set_tests_properties(contract_rejects_unknown PROPERTIES WILL_FAIL TRUE)
if(TARGET aar_mujoco_smoke)
  add_test(
    NAME pe01_mujoco_3_8_load_step
    COMMAND aar_mujoco_smoke
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/unilab/assets/robots/pe01/scene.xml
  )
  set_tests_properties(pe01_mujoco_3_8_load_step PROPERTIES PASS_REGULAR_EXPRESSION "MuJoCo 3.8.0")
endif()
if(TARGET aar_sim2sim AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../releases/examples/pe01_flat/policy.onnx)
  foreach(AAR_TRAJECTORY_STEPS 1 10 100)
    add_test(
      NAME pe01_onnx_sim2sim_${AAR_TRAJECTORY_STEPS}_steps
      COMMAND aar_sim2sim
        ${CMAKE_CURRENT_SOURCE_DIR}/../releases/examples/pe01_flat
        --steps ${AAR_TRAJECTORY_STEPS}
        --telemetry ${CMAKE_CURRENT_BINARY_DIR}/telemetry-${AAR_TRAJECTORY_STEPS}
    )
    set_tests_properties(pe01_onnx_sim2sim_${AAR_TRAJECTORY_STEPS}_steps
      PROPERTIES PASS_REGULAR_EXPRESSION "sim2sim ok robot=pe01")
  endforeach()
endif()
