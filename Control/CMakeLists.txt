# @author Isaac Jesus da Silva - ROBOFEI-HT - FEI
# @version V0.0.3
# @created 04/07/2014
# @Modified 01/06/2016
# 
# Para Help acesse http://www.cmake.org/cmake/help/v2.8.0/cmake.html#command:link_directories

find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(
  include
  ../Blackboard/include
  ../Control/Linux/include
  ../Control/Framework/include
  ../Control/Linux/project/control
)

include_directories(include)
add_library(libcontrol SHARED ./Linux/project/control/main.cpp ./Linux/project/control/kbhit.cpp ./Linux/project/control/ReadConfig.cpp ./Linux/project/control/ActionMove.cpp ./Linux/project/control/GaitMove.cpp)
install(TARGETS libcontrol DESTINATION ~/RoboFEI-DNN/build/lib)


#cria executável
add_executable(control ./Linux/project/control/main.cpp)

#adiciona as dependências
add_dependencies(control libcontrol blackboard)

set(LIB_DARWIN ~/RoboFEI-DNN/Control/Linux/lib/darwin.a)

target_link_libraries(control libcontrol blackboard ${LIB_DARWIN} -Wno-write-strings -lpthread -lrt ${Boost_LIBRARIES})


