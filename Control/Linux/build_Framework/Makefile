###############################################################
#
# Purpose: Makefile for "DARwIn Linux Framework"
# Author.: robotis
# Version: 0.1
# License: GPL
#
###############################################################

CC = g++
AR = ar
ARFLAGS = cr

TARGET = darwin.a

INCLUDE_DIRS = -I../include -I../../Framework/include -I../../../Blackboard/include

CXXFLAGS += -O2 -DLINUX -D_GNU_SOURCE -Wall -g -shared $(INCLUDE_DIRS) 
#CXXFLAGS += -O2 -DDEBUG -DLINUX -D_GNU_SOURCE -Wall -shared $(INCLUDE_DIRS)
LFLAGS += -g -lpthread -ldl -lbluetooth

OBJS =  ../../Framework/src/CM730.o     	\
        ../../Framework/src/math/Matrix.o   \
        ../../Framework/src/math/Plane.o    \
        ../../Framework/src/math/Point.o    \
        ../../Framework/src/math/Vector.o   \
		../../Framework/src/math/QuadraticStateTransform.o   \
		../../Framework/src/math/MotionState.o   \
        ../../Framework/src/motion/JointData.o  	\
        ../../Framework/src/motion/Kinematics.o 	\
        ../../Framework/src/motion/MotionManager.o  \
        ../../Framework/src/motion/MotionStatus.o   \
		../../Framework/src/motion/AngleEstimator.o \
        ../../Framework/src/motion/modules/Action.o \
        ../../Framework/src/motion/modules/Walking.o\
        ../../Framework/src/minIni/minIni.o	\
        LinuxActionScript.o   \
        LinuxCM730.o    \
        LinuxMotionTimer.o    \
        LinuxNetwork.o

$(TARGET): $(OBJS)
	mkdir -p ../lib/
	$(AR) $(ARFLAGS) ../lib/$(TARGET) $(OBJS) 
#	$(CC) -o $(TARGET) $(OBJS) $(LFLAGS)

all: $(TARGET)

clean:
	rm -f $(OBJS) ../lib/$(TARGET)

