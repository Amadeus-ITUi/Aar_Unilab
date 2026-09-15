cmake_minimum_required(VERSION 3.16)
project(aar_unilab_sim2sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(aar_runtime STATIC
  src/deployment_contract.cpp
  src/telemetry.cpp
)
target_include_directories(aar_runtime PUBLIC include)
target_compile_options(aar_runtime PRIVATE -Wall -Wextra -Wpedantic)

add_executable(aar_contract_check src/main.cpp)
target_link_libraries(aar_contract_check PRIVATE aar_runtime)

set(AAR_MUJOCO_ROOT "" CACHE PATH "MuJoCo Python package root containing include/ and libmujoco")
if(AAR_MUJOCO_ROOT)
  find_path(AAR_MUJOCO_INCLUDE mujoco/mujoco.h PATHS ${AAR_MUJOCO_ROOT}/include NO_DEFAULT_PATH REQUIRED)
  find_library(AAR_MUJOCO_LIBRARY NAMES libmujoco.so.3.8.0 mujoco PATHS ${AAR_MUJOCO_ROOT} NO_DEFAULT_PATH REQUIRED)
  add_executable(aar_mujoco_smoke src/mujoco_smoke.cpp)
  target_include_directories(aar_mujoco_smoke PRIVATE ${AAR_MUJOCO_INCLUDE})
  target_link_libraries(aar_mujoco_smoke PRIVATE ${AAR_MUJOCO_LIBRARY})
  set_target_properties(aar_mujoco_smoke PROPERTIES BUILD_RPATH ${AAR_MUJOCO_ROOT})
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
