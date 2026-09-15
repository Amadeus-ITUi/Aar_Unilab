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
