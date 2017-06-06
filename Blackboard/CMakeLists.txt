include_directories(include)
add_library(blackboard SHARED src/blackboard.cpp)
file(COPY src/blackboard.cpp  DESTINATION ../../Blackboard)
file(RENAME blackboard.cpp ../build/Blackboard/blackboard.c)
add_library(blackboardpy SHARED ../Blackboard/blackboard.c)
install(TARGETS blackboard DESTINATION ~/RoboFEI-DNN/build/lib)
install(TARGETS blackboardpy DESTINATION ~/RoboFEI-DNN/build/lib)


