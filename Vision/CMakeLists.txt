
include_directories(
	include
	./Dynamixel/include
    ../Blackboard/include
)


add_library(vision SHARED ./src/vision.cpp ./Dynamixel/src/dxl_hal.c ./Dynamixel/src/dynamixel.c)
install(TARGETS vision DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib)


