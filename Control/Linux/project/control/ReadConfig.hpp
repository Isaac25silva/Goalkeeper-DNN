/*--------------------------------------------------------------------

******************************************************************************
* @file mov.hpp
* @author Vinicius Nicassio Ferreira - ROBOFEI-HT - FEI 😛
* @version V1.0.5
* @created 5/01/2016
* @Modified 5/01/2016
* @e-mail vinicius.nicassio@gmail.com
* @revisor Isaac Jesus da Silva
* @e-mail isaac25silva@yahoo.com.br
* @brief mov 😛
****************************************************************************
**************************************************************************** 
Arquivo cabeçalho contendo metodos de movimentos de acoes e gerador de caminhada
/--------------------------------------------------------------------------*/

#ifndef ReadConfig_H
#define ReadConfig_H

#include <string>
#include "minIni.h"
#include "Walking.h"

using namespace Robot;

// Criação da Classe
class ReadConfig
{
		public:
		
        double turn_angle;
        double walk_foward;
        double sidle;

		//Construtor
		ReadConfig(std::string a, minIni* ini);
		
		//Destrutor
		~ReadConfig(){};
    	
    	void changeParam(Walking* walk);
    	
    	private:
        double x_offset;
        double y_offset;
        double z_offset;
        double roll_offset;
        double pitch_offset;
        double yaw_offset;
        double hip_pitch_offset;

        double period_time;
        double dsp_ratio;
        double step_forward_back_ratio;
        double foot_height;
        double swing_right_left;
        double swing_top_down;
        double pelvis_offset;
        double arm_swing_gain;

        double balance_knee_gain;
        double balance_ankle_pitch_gain;
        double balance_hip_roll_gain;
        double balance_ankle_roll_gain;
        double balance_angle_smooth_gain;
        double balance_angle_gain;
        double lean_fb_gain;
        double lean_fb_accel_gain;
        double lean_turn_gain;
        double start_step_factor;
        double p_gain;
        double i_gain;
        double d_gain;
};

//*********************************************************************
//---------------------------------------------------------------------
#endif
