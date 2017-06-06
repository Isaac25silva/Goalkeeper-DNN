#coding: utf-8
 # ----------------------------------------------------------------------------
 # ****************************************************************************
 # * @file SharedMemory.py
 # * @author Isaac Jesus da Silva - ROBOFEI-HT - FEI 😛
 # * @version V0.0.2
 # * @created 08/09/2015
 # * @Modified 16/09/2015
 # * @e-mail isaac25silva@yahoo.com.br
 # * @brief Shared Memory
 # ****************************************************************************
 # ****************************************************************************
import ctypes
import time
import os


# Classe do BlackBoard--------------------------------------------------------------------
class SharedMemory(object):
# Classe que lê e escreve na memória compartilhada do sistema '''

    def __init__(self):
        #print "Start the Class Blackboard"
        # Usando memoria compartilhada a partir das funções do c++-------------------------------------------------------
        try:
            self.testlib = ctypes.CDLL('../../build/lib/libblackboardpy.so') #chama a library que contem as funções em c++
        except OSError:
            self.testlib = ctypes.CDLL('../AI/build/lib/libblackboardpy.so') #chama a library que contem as funções em c++
        self.testlib.using_shared_memory.restype = ctypes.POINTER(ctypes.c_int)


    def shd_constructor(self,KEY):
        mem = self.testlib.using_shared_memory(KEY)         #using c++ function
        #print 'python', mem
        self.testlib.read_float.restype = ctypes.c_float #defining the return type, that case defining float
        self.testlib.read_int.restype = ctypes.c_int #defining the return type, that case defining int
        return mem
        #--------------------------------------------------------------------------------------------------------------------


    # Criando função que escreve float--------------------------------------------------------
    def write_float(self, mem, variable, value):
        self.testlib.write_float(mem, ctypes.c_int(self.variable_float[variable]), ctypes.c_float(value))
    #-----------------------------------------------------------------------------------------


    # Criando função que escreve float--------------------------------------------------------
    def write_floatDynamic(self, mem, variable,index, value):
        self.testlib.write_float(mem, ctypes.c_int(self.variable_float[variable]+index), ctypes.c_float(value))
    #-----------------------------------------------------------------------------------------

    # Criando função que escreve float--------------------------------------------------------
    def write_int(self, mem, variable, value):
        self.testlib.write_int(mem, ctypes.c_int(self.variable_int[variable]), ctypes.c_int(int(value)))
    #-----------------------------------------------------------------------------------------

    # Criando função que lê float--------------------------------------------------------
    def read_float(self, mem, variable):
        return self.testlib.read_float(mem, ctypes.c_int(self.variable_float[variable]))
    #-----------------------------------------------------------------------------------------

    # Criando função que lê float--------------------------------------------------------
    def read_int(self, mem, variable):
        return self.testlib.read_int(mem, ctypes.c_int(self.variable_int[variable]))
    #-----------------------------------------------------------------------------------------

    variable_int = {
    'PLANNING_COMMAND' : 0,
    'PLANNING_PARAMETER_VEL': 1,
    'PLANNING_PARAMETER_ANGLE': 2,
    'IMU_STATE' : 3,
    'IMU_RESET': 4,
    'CONTROL_ACTION': 13,
    'CONTROL_HEIGHT_A': 14,
    'CONTROL_HEIGHT_B': 15,
    'CONTROL_HEIGHT_C': 16,
    'DECISION_ACTION_A': 17,
    'DECISION_ACTION_B': 18,
    'DECISION_STATE': 19,
    'DECISION_POSITION_A': 20,
    'DECISION_POSITION_B': 21,
    'DECISION_POSITION_C': 22,
    'DECISION_BALL_POS': 23,
    'DECISION_OPP1_POS': 24,
    'DECISION_OPP2_POS': 25,
    'DECISION_OPP3_POS': 26,
    'COM_ACTION_ROBOT1': 27,
    'COM_ACTION_ROBOT2': 28,
    'COM_ACTION_ROBOT3': 29,
    'COM_STATE_ROBOT1': 30,
    'COM_STATE_ROBOT2': 31,
    'COM_STATE_ROBOT3': 32,
    'COM_POS_ROBOT1': 33,
    'COM_POS_ROBOT2': 34,
    'COM_POS_ROBOT3': 35,
    'COM_POS_BALL_ROBOT1': 36,
    'COM_POS_BALL_ROBOT2': 37,
    'COM_POS_BALL_ROBOT3': 38,
    'COM_POS_OPP_A_ROBOT1': 39,
    'COM_POS_OPP_A_ROBOT2': 40,
    'COM_POS_OPP_A_ROBOT3': 41,
    'COM_POS_OPP_A_ROBOT4': 42,
    'COM_POS_OPP_B_ROBOT1': 43,
    'COM_POS_OPP_B_ROBOT2': 44,
    'COM_POS_OPP_B_ROBOT3': 45,
    'COM_POS_OPP_B_ROBOT4': 46,
    'COM_POS_OPP_C_ROBOT1': 47,
    'COM_POS_OPP_C_ROBOT2': 48,
    'COM_POS_OPP_C_ROBOT3': 49,
    'COM_POS_OPP_C_ROBOT4': 50,
    'COM_REFEREE': 51,
    'LOCALIZATION_X': 52,
    'LOCALIZATION_Y': 53,
    'LOCALIZATION_THETA': 54,
    'VISION_LOST': 57,
    'DECISION_SEARCH_ON': 58,
    'DECISION_ACTION_VISION': 59,
    'VISION_MOTOR1_GOAL': 60,
    'VISION_MOTOR2_GOAL': 61,
    'VISION_SEARCH_GOAL': 62,
    'VISION_LOST_GOAL': 63,
    'VISION_STATE': 64,
    'ROBOT_NUMBER': 65,
    'VISION_pos_servo1': 66,
    'VISION_pos_servo2': 67,
    'COM_POS_ORIENT_QUALIT_ROBOT_A': 68,
    'COM_POS_DIST_QUALIT_ROBOT_A': 69,
    'COM_POS_ORIENT_QUALIT_ROBOT_B': 70,
    'COM_POS_DIST_QUALIT_ROBOT_B': 71,
    'COM_POS_ORIENT_QUALIT_ROBOT_C': 72,
    'COM_POS_DIST_QUALIT_ROBOT_C': 73,
    'VISION_DELTA_ORIENT': 74,
    'LOCALIZATION_FIND_ROBOT': 75,
    'RECEIVED_ROBOT_SENDING': 76,
    'RECEIVED_QUAL_ORIENT': 77,
    'RECEIVED_QUAL_DIST': 78,
    'RECEIVED_ROBOT_SEEN': 79,
    'CONTROL_MESSAGES': 80,
    'ASKED_QUALIT_DIRECT': 81,
    'ASKED_QUALIT_DISTANCE': 82,
    'ASKED_RELATED_ROBOT': 83,
    'CONTROL_MOVING': 84,
    'ROBOT_VIEW_ROTATE': 100,
    'CONTROL_WORKING': 101,
    'VISION_WORKING': 102,
    'LOCALIZATION_WORKING': 103,
    'DECISION_WORKING': 104,
    'IMU_WORKING': 105,
    'VOLTAGE': 106,
    'DECISION_LOCALIZATION': 107,
    }

    variable_float = {
    'IMU_GYRO_X': 1,
    'IMU_GYRO_Y': 2,
    'IMU_GYRO_Z': 3,
    'IMU_ACCEL_X': 4,
    'IMU_ACCEL_Y': 5,
    'IMU_ACCEL_Z': 6,
    'IMU_COMPASS_X': 7,
    'IMU_COMPASS_Y': 8,
    'IMU_COMPASS_Z': 9,
    'IMU_EULER_X': 10,
    'IMU_EULER_Y': 11,
    'IMU_EULER_Z': 12,
    'IMU_QUAT_X': 13,
    'IMU_QUAT_Y': 14,
    'IMU_QUAT_Z': 15,
    'VISION_AREA_SEGMENT': 16,
    'VISION_BALL_DIST': 17,
    'VISION_BALL_ANGLE': 18,
    'VISION_GOAL_DIST': 19,
    'VISION_GOAL_ANGLE': 20,
    'VISION_OPP01_DIST': 21,
    'VISION_OPP02_DIST': 22,
    'VISION_OPP03_DIST': 23,
    'VISION_OPP04_DIST': 24,
    'VISION_OPP05_DIST': 25,
    'VISION_OPP06_DIST': 26,
    'VISION_OPP07_DIST': 27,
    'VISION_OPP08_DIST': 28,
    'VISION_OPP09_DIST': 29,
    'VISION_OPP10_DIST': 30,
    'VISION_OPP11_DIST': 31,
    'VISION_OPP01_ANGLE': 32,
    'VISION_OPP02_ANGLE': 33,
    'VISION_OPP03_ANGLE': 34,
    'VISION_OPP04_ANGLE': 35,
    'VISION_OPP05_ANGLE': 36,
    'VISION_OPP06_ANGLE': 37,
    'VISION_OPP07_ANGLE': 38,
    'VISION_OPP08_ANGLE': 39,
    'VISION_OPP09_ANGLE': 40,
    'VISION_OPP10_ANGLE': 41,
    'VISION_OPP11_ANGLE': 42,
    'VISION_RBT01_DIST': 43,
    'VISION_RBT02_DIST': 44,
    'VISION_RBT03_DIST': 45,
    'VISION_RBT04_DIST': 46,
    'VISION_RBT05_DIST': 47,
    'VISION_RBT06_DIST': 48,
    'VISION_RBT07_DIST': 49,
    'VISION_RBT08_DIST': 50,
    'VISION_RBT09_DIST': 51,
    'VISION_RBT10_DIST': 52,
    'VISION_RBT11_DIST': 53,
    'VISION_RBT01_ANGLE': 54,
    'VISION_RBT02_ANGLE': 55,
    'VISION_RBT03_ANGLE': 56,
    'VISION_RBT04_ANGLE': 57,
    'VISION_RBT05_ANGLE': 58,
    'VISION_RBT06_ANGLE': 59,
    'VISION_RBT07_ANGLE': 60,
    'VISION_RBT08_ANGLE': 61,
    'VISION_RBT09_ANGLE': 62,
    'VISION_RBT10_ANGLE': 63,
    'VISION_RBT11_ANGLE': 64,
    'VISION_TILT_DEG': 65,
    'VISION_PAN_DEG': 66,
    'CBR_COORDINATOR': 67,
    'CBR_RUN': 68,
    'LOCALIZATION_BALL_X': 69,
    'LOCALIZATION_BALL_Y': 70,
    'LOCALIZATION_RBT01_X': 71,
    'LOCALIZATION_RBT01_Y': 72,
    'LOCALIZATION_RBT02_X': 73,
    'LOCALIZATION_RBT02_Y': 74,
    'LOCALIZATION_RBT03_X': 75,
    'LOCALIZATION_RBT03_Y': 76,
    'LOCALIZATION_RBT04_X': 77,
    'LOCALIZATION_RBT04_Y': 78,
    'LOCALIZATION_RBT05_X': 79,
    'LOCALIZATION_RBT05_Y': 80,
    'LOCALIZATION_RBT06_X': 81,
    'LOCALIZATION_RBT06_Y': 82,
    'LOCALIZATION_RBT07_X': 83,
    'LOCALIZATION_RBT07_Y': 84,
    'LOCALIZATION_RBT08_X': 85,
    'LOCALIZATION_RBT08_Y': 86,
    'LOCALIZATION_RBT09_X': 87,
    'LOCALIZATION_RBT09_Y': 88,
    'LOCALIZATION_RBT10_X': 89,
    'LOCALIZATION_RBT10_Y': 90,
    'LOCALIZATION_RBT11_X': 91,
    'LOCALIZATION_RBT11_Y': 92,
    'LOCALIZATION_OPP01_X': 93,
    'LOCALIZATION_OPP01_Y': 94,
    'LOCALIZATION_OPP02_X': 95,
    'LOCALIZATION_OPP02_Y': 96,
    'LOCALIZATION_OPP03_X': 97,
    'LOCALIZATION_OPP03_Y': 98,
    'LOCALIZATION_OPP04_X': 99,
    'LOCALIZATION_OPP04_Y': 100,
    'LOCALIZATION_OPP05_X': 101,
    'LOCALIZATION_OPP05_Y': 102,
    'LOCALIZATION_OPP06_X': 103,
    'LOCALIZATION_OPP06_Y': 104,
    'LOCALIZATION_OPP07_X': 105,
    'LOCALIZATION_OPP07_Y': 106,
    'LOCALIZATION_OPP08_X': 107,
    'LOCALIZATION_OPP08_Y': 108,
    'LOCALIZATION_OPP09_X': 109,
    'LOCALIZATION_OPP09_Y': 110,
    'LOCALIZATION_OPP10_X': 111,
    'LOCALIZATION_OPP10_Y': 112,
    'LOCALIZATION_OPP11_X': 113,
    'LOCALIZATION_OPP11_Y': 114,
    'DECISION_RBT01_DIST_BALL': 115,
    'DECISION_RBT02_DIST_BALL': 116,
    'DECISION_RBT03_DIST_BALL': 117,
    'DECISION_RBT04_DIST_BALL': 118,
    'VISION_BLUE_LANDMARK_DEG': 119,
    'VISION_RED_LANDMARK_DEG': 120,
    'VISION_YELLOW_LANDMARK_DEG': 121,
    'VISION_PURPLE_LANDMARK_DEG': 122,
    }
#------------------------------------------------------------------------------------------
