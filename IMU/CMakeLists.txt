# @author Isaac Jesus da Silva - ROBOFEI-HT - FEI
# @version V0.0.1
# @created 24/08/2015
# @Modified 01/06/2017 
#
# 
# Para Help acesse http://www.cmake.org/cmake/help/v2.8.0/cmake.html#command:link_directories

find_package( Boost REQUIRED )

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ../Blackboard/include
  ../IMU/include/um7
  ./serial/include

)

include_directories(include)
add_library(comms SHARED src/comms.cpp)
install(TARGETS comms DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib)
add_library(registers SHARED src/registers.cpp)
install(TARGETS registers DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib)


add_executable(imu src/imu.cpp src/INIReader.cpp src/ini.c)
add_dependencies(imu 
blackboard serial comms registers ${Boost_LIBRARIES})

target_link_libraries(imu 
blackboard serial comms registers rt pthread ${Boost_LIBRARIES})



