cmake_minimum_required(VERSION 3.8)
project(octomap_update)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(octomap REQUIRED)
find_package(rclcpp REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(octomap_update src/octomap_update.cpp)
target_include_directories(octomap_update PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(octomap_update PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(octomap_update
  "rclcpp"
  "octomap"
  "octomap_msgs"
  "sensor_msgs"
  "std_msgs"
  "geometry_msgs"
)

install(TARGETS octomap_update
  DESTINATION lib/${PROJECT_NAME})

# config and launch files
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/)

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

ament_package()
