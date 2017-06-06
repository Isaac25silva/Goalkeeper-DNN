/*--------------------------------------------------------------------

******************************************************************************
* @file GaitMove.hpp
* @author Isaac Jesus da Silva - ROBOFEI-HT - FEI 😛
* @version V1.1.0
* @created 15/07/2016
* @Modified 15/07/2016
* @e-mail isaac25silva@yahoo.com.br
* @revisor Isaac Jesus da Silva
* @e-mail isaac25silva@yahoo.com.br
* @brief giat 😛
****************************************************************************
**************************************************************************** 
Arquivo cabeçalho contendo metodos de movimentos do gait
/--------------------------------------------------------------------------*/

#ifndef GaitMove_H
#define GaitMove_H

#include "Action.h"
#include "Walking.h"
#include "LinuxCM730.h"
#include "minIni.h"
#include "ReadConfig.hpp"
#include <blackboard.h>

using namespace Robot;

// Criação da Classe
class GaitMove
{
    public:

        //Construtor
        GaitMove(int *mem_t, minIni* ini);

        //Destrutor
        ~GaitMove(){};

        void Gait_in_place(bool &stop_gait, bool same_moviment);

        void walk_foward_slow(bool &stop_gait, bool max_speed, bool same_moviment, bool &enable_soft_starter);

        void robot_stop(bool &stop_gait);

        void walk_foward_fast_direct(bool &stop_gait, bool same_moviment);

        void walk_foward_fast(bool &stop_gait, bool same_moviment, bool &enable_soft_starter);

        void turn_right(bool &stop_gait, bool max_angle, bool same_moviment);

        void turn_left(bool &stop_gait, bool max_angle, bool same_moviment);

        void sidle_right(bool &stop_gait, bool same_moviment);

        void sidle_left(bool &stop_gait, bool same_moviment);

        void turn_around_ball_right(bool &stop_gait, bool same_moviment);

        void turn_around_ball_left(bool &stop_gait, bool same_moviment);

        void walk_backward_fast(bool &stop_gait, bool same_moviment);

        void walk_backward_slow(bool &stop_gait, bool max_speed, bool same_moviment);

    private:
        int *mem;
        double soft_starter_value;
        ReadConfig* gait;
        ReadConfig* walkfoward;
        ReadConfig* turnRobotLeft;
        ReadConfig* turnRobotRight;
        ReadConfig* walkslow;
        ReadConfig* turnBallR;
        ReadConfig* turnBallL;
        ReadConfig* sidleR;
        ReadConfig* sidleL;
        ReadConfig* walkbackward;
        ReadConfig* walkbackwardslow;

       void move_gait(float X_amplitude, float Y_amplitude, float A_amplitude, bool &stop_gait, ReadConfig* configP);

};

//*********************************************************************
//---------------------------------------------------------------------
#endif
