cmake_minimum_required(VERSION 3.8)
project(pinchy_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes DESTINATION share/${PROJECT_NAME})
install(FILES THIRD_PARTY.md LICENSE-SO-ARM100 DESTINATION share/${PROJECT_NAME})

ament_package()
