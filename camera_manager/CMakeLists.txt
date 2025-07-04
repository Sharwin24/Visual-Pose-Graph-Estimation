cmake_minimum_required(VERSION 3.8)
project(camera_manager)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenGL_GL_PREFERENCE GLVND)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV 4.6 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nuturtlebot_msgs REQUIRED)
find_package(std_srvs REQUIRED)


include_directories(
  include/camera_manager
  ${OpenCV_INCLUDE_DIRS}
  ${g2o_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)


# Camera Manager Node
add_executable(CAMERA_MANAGER src/camera_manager.cpp)
ament_target_dependencies(CAMERA_MANAGER
  rclcpp
  sensor_msgs
  OpenCV
  cv_bridge
  image_transport
  nav_msgs
  tf2_ros
  message_filters
  geometry_msgs
  visualization_msgs
)
target_link_libraries(CAMERA_MANAGER
  ${OpenCV_LIBRARIES}
  g2o_core
  g2o_stuff
  g2o_types_slam3d
  g2o_solver_csparse
  g2o_solver_dense
)
install(TARGETS
CAMERA_MANAGER
  DESTINATION lib/${PROJECT_NAME}
)

# Path Creator Node
add_executable(path_creator src/path_creator.cpp)
ament_target_dependencies(path_creator
  rclcpp
  sensor_msgs
  nav_msgs
  tf2_ros
  geometry_msgs
  visualization_msgs
  std_srvs
)

# Teleop node
install(PROGRAMS
  camera_manager/teleop.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  path_creator
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)

  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_python_install_package(${PROJECT_NAME})
ament_package()