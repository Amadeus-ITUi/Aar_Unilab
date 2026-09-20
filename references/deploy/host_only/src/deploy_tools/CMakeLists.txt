cmake_minimum_required(VERSION 3.8)
project(deploy_tools)

find_package(ament_cmake REQUIRED)

function(install_deploy_tool source_path)
  get_filename_component(executable_name "${source_path}" NAME_WE)
  install(
    PROGRAMS "${source_path}"
    DESTINATION lib/${PROJECT_NAME}
    RENAME "${executable_name}"
  )
endfunction()

set(DEPLOY_TOOL_SCRIPTS
  scripts/can/decode_motor_can.py
  scripts/can/read_all_motor_status.py
  scripts/can/read_calf_status_0x02.py
  scripts/can/read_motor_firmware.py
  scripts/can/read_motor_status.py
  scripts/diagnostics/check_imu_data.py
  scripts/diagnostics/read_only_motor_state_node.py
  scripts/diagnostics/record_deploy_runtime.py
  scripts/diagnostics/record_policy_imu_sequence.py
  scripts/diagnostics/summarize_standing_observation.py
  scripts/control/set_foot_pd.py
  scripts/control/emergency_disable_motors.py
  scripts/control/we11_gamepad_launcher.py
  scripts/control/we11_state.py
  scripts/control/we11_supervisor.py
  scripts/hardware/rgb_led.py
  scripts/hardware/rgb_led_cycle.py
  scripts/analysis/plot_calf_command_position_torque.py
  scripts/analysis/plot_motor_2_5_comparison.py
  scripts/analysis/plot_policy_output_vs_joint_state.py
  scripts/analysis/plot_policy_pair_tracking.py
)

foreach(tool_script IN LISTS DEPLOY_TOOL_SCRIPTS)
  install_deploy_tool("${tool_script}")
endforeach()

install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(we11_state_test test/test_we11_state.py)
  ament_add_pytest_test(we11_gamepad_launcher_test test/test_we11_gamepad_launcher.py)
  ament_add_pytest_test(rgb_led_test test/test_rgb_led.py)
  ament_add_pytest_test(we11_supervisor_ros_test test/test_we11_supervisor_ros.py
    TIMEOUT 30)
endif()

ament_package()
