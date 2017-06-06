#include_directories(${RobotFEI-HT_SOURCE_DIR}/Blackboard/include)
include_directories(include)
add_library(dxl_hal SHARED src/dxl_hal.c)
install(TARGETS dxl_hal DESTINATION ${RoboFEI-HT_BINARY_DIR}/lib)
add_library(dynamixel SHARED src/dynamixel.c)	 
install(TARGETS dynamixel DESTINATION ${RoboFEI-HT_BINARY_DIR}/lib)
#install(TARGETS blackboard DESTINATION /usr/local/lib)



