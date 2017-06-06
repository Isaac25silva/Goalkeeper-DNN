cmake_minimum_required(VERSION 2.8.3)
project(serial)

include_directories(
  include
  ${serial_SOURCE_DIR}/include
  ${serial_SOURCE_DIR}/include/impl
)
 

## Sources
set(serial_SRCS
    src/serial.cc
    include/serial.h
    include/v8stdint.h
   
)
if(APPLE)
	# If OSX
	list(APPEND serial_SRCS src/impl/unix.cc)
	list(APPEND serial_SRCS src/impl/list_ports/list_ports_osx.cc)
elseif(UNIX)
    # If unix
    list(APPEND serial_SRCS src/impl/unix.cc)
    list(APPEND serial_SRCS src/impl/list_ports/list_ports_linux.cc)
elseif()
    # If windows
    list(APPEND serial_SRCS src/impl/win.cc)
endif()

## Add serial library
add_library(${PROJECT_NAME} ${serial_SRCS})
if(APPLE)
	target_link_libraries(${PROJECT_NAME} ${FOUNDATION_LIBRARY} ${IOKIT_LIBRARY}) 
elseif(UNIX)
	target_link_libraries(${PROJECT_NAME} rt pthread)
endif()

## Uncomment for example
add_executable(serial_example examples/serial_example.cc)
add_dependencies(serial_example ${PROJECT_NAME})
target_link_libraries(serial_example ${PROJECT_NAME})

## Include headers
include_directories(include)

## Install executable
install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION /usr/local/lib
    LIBRARY DESTINATION /usr/local/lib
)

## Install headers
install(FILES include/serial.h include/v8stdint.h
  DESTINATION /usr/local/include/serial)

