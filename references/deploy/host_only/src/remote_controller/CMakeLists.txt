cmake_minimum_required(VERSION 3.8)
project(remote_controller)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR src/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/remote_controller_node
  scripts/joy_cmd_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
