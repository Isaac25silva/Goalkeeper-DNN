#include <iostream>
#include <stdio.h>
#include "ReadConfig.hpp"

ReadConfig::ReadConfig(std::string a, minIni* ini)
{
	if((x_offset=ini->getd(a,"x_offset",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" x_offset"<<std::endl;
		if((x_offset=ini->getd("Walking Config","x_offset",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config x_offset, usando valor padrão"<<std::endl;
			x_offset=-15;
		}
	}
			
	if((y_offset=ini->getd(a,"y_offset",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" y_offset"<<std::endl;
		if((y_offset=ini->getd("Walking Config","y_offset",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config y_offset, usando valor padrão"<<std::endl;
			y_offset=5;
		}
	}
			
	if((z_offset=ini->getd(a,"z_offset",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" z_offset"<<std::endl;
		if((z_offset=ini->getd("Walking Config","z_offset",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config z_offset, usando valor padrão"<<std::endl;
			z_offset=20;
		}
	}
			
	if((roll_offset=ini->getd(a,"roll_offset",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" roll_offset"<<std::endl;
		if((roll_offset=ini->getd("Walking Config","roll_offset",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config roll_offset, usando valor padrão"<<std::endl;
			roll_offset=0;
		}
	}
			
	if((pitch_offset=ini->getd(a,"pitch_offset",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" pitch_offset"<<std::endl;
		if((pitch_offset=ini->getd("Walking Config","pitch_offset",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config pitch_offset, usando valor padrão"<<std::endl;
			pitch_offset=0;
		}
	}
			
	if((yaw_offset=ini->getd(a,"yaw_offset",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" yaw_offset"<<std::endl;
		if((yaw_offset=ini->getd("Walking Config","yaw_offset",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config yaw_offset, usando valor padrão"<<std::endl;
			yaw_offset=0;
		}
	}
			
	if((hip_pitch_offset=ini->getd(a,"hip_pitch_offset",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" hip_pitch_offset"<<std::endl;
		if((hip_pitch_offset=ini->getd("Walking Config","hip_pitch_offset",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config hip_pitch_offset, usando valor padrão"<<std::endl;
			hip_pitch_offset=13;
		}
	}
			
	if((period_time=ini->getd(a,"period_time",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" period_time"<<std::endl;
		if((period_time=ini->getd("Walking Config","period_time",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config period_time, usando valor padrão"<<std::endl;
			period_time=600;
		}
	}
			
	if((dsp_ratio=ini->getd(a,"dsp_ratio",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" dsp_ratio"<<std::endl;
		if((dsp_ratio=ini->getd("Walking Config","dsp_ratio",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config dsp_ratio, usando valor padrão"<<std::endl;
			dsp_ratio=0.1;
		}
	}
			
	if((step_forward_back_ratio=ini->getd(a,"step_forward_back_ratio",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" step_forward_back_ratio"<<std::endl;
		if((step_forward_back_ratio=ini->getd("Walking Config","step_forward_back_ratio",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config step_forward_back_ratio, usando valor padrão"<<std::endl;
			step_forward_back_ratio=0.21;
		}
	}
	if((foot_height=ini->getd(a,"foot_height",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" foot_height"<<std::endl;
		if((foot_height=ini->getd("Walking Config","foot_height",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config foot_height, usando valor padrão"<<std::endl;
			foot_height=40;
		}
	}
			
	if((swing_right_left=ini->getd(a,"swing_right_left",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" swing_right_left"<<std::endl;
		if((swing_right_left=ini->getd("Walking Config","swing_right_left",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config swing_right_left, usando valor padrão"<<std::endl;
			swing_right_left=20;
		}
	}
			
	if((swing_top_down=ini->getd(a,"swing_top_down",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" swing_top_down"<<std::endl;
		if((swing_top_down=ini->getd("Walking Config","swing_top_down",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config swing_top_down, usando valor padrão"<<std::endl;
			swing_top_down=5;
		}
	}
			
	if((pelvis_offset=ini->getd(a,"pelvis_offset",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" pelvis_offset"<<std::endl;
		if((pelvis_offset=ini->getd("Walking Config","pelvis_offset",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config pelvis_offset, usando valor padrão"<<std::endl;
			pelvis_offset=3;
		}
	}
			
	if((arm_swing_gain=ini->getd(a,"arm_swing_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" arm_swing_gain"<<std::endl;
		if((arm_swing_gain=ini->getd("Walking Config","arm_swing_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config arm_swing_gain, usando valor padrão"<<std::endl;
			arm_swing_gain=1.5;
		}
	}
			
	if((balance_knee_gain=ini->getd(a,"balance_knee_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" balance_knee_gain"<<std::endl;
		if((balance_knee_gain=ini->getd("Walking Config","balance_knee_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config balance_knee_gain, usando valor padrão"<<std::endl;
			balance_knee_gain=0.3;
		}
	}
			
	if((balance_ankle_pitch_gain=ini->getd(a,"balance_ankle_pitch_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" balance_ankle_pitch_gain"<<std::endl;
		if((balance_ankle_pitch_gain=ini->getd("Walking Config","balance_ankle_pitch_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config balance_ankle_pitch_gain, usando valor padrão"<<std::endl;
			balance_ankle_pitch_gain=0.9;
		}
	}
			
	if((balance_hip_roll_gain=ini->getd(a,"balance_hip_roll_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" balance_hip_roll_gain"<<std::endl;
		if((balance_hip_roll_gain=ini->getd("Walking Config","balance_hip_roll_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config balance_hip_roll_gain, usando valor padrão"<<std::endl;
			balance_hip_roll_gain=0.5;
		}
	}
			
	if((balance_ankle_roll_gain=ini->getd(a,"balance_ankle_roll_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" balance_ankle_roll_gain"<<std::endl;
		if((balance_ankle_roll_gain=ini->getd("Walking Config","balance_ankle_roll_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config balance_ankle_roll_gain, usando valor padrão"<<std::endl;
			balance_ankle_roll_gain=0.1;
		}
	}
			
	if((balance_angle_smooth_gain=ini->getd(a,"balance_angle_smooth_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" balance_angle_smooth_gain"<<std::endl;
		if((balance_angle_smooth_gain=ini->getd("Walking Config","balance_angle_smooth_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config balance_angle_smooth_gain, usando valor padrão"<<std::endl;
			balance_angle_smooth_gain=0.91;
		}
	}
			
	if((balance_angle_gain=ini->getd(a,"balance_angle_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" balance_angle_gain"<<std::endl;
		if((balance_angle_gain=ini->getd("Walking Config","balance_angle_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config balance_angle_gain, usando valor padrão"<<std::endl;
			balance_angle_gain=0.1;
		}
	}
			
	if((lean_fb_gain=ini->getd(a,"lean_fb_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" lean_fb_gain"<<std::endl;
		if((lean_fb_gain=ini->getd("Walking Config","lean_fb_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config lean_fb_gain, usando valor padrão"<<std::endl;
			lean_fb_gain=0;
		}
	}
			
	if((lean_fb_accel_gain=ini->getd(a,"lean_fb_accel_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" lean_fb_accel_gain"<<std::endl;
		if((lean_fb_accel_gain=ini->getd("Walking Config","lean_fb_accel_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config lean_fb_accel_gain, usando valor padrão"<<std::endl;
			lean_fb_accel_gain=0;
		}
	}
			
	if((lean_turn_gain=ini->getd(a,"lean_turn_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" lean_turn_gain"<<std::endl;
		if((lean_turn_gain=ini->getd("Walking Config","lean_turn_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config lean_turn_gain, usando valor padrão"<<std::endl;
			lean_turn_gain=0;
		}
	}
			
	if((start_step_factor=ini->getd(a,"start_step_factor",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" start_step_factor"<<std::endl;
		if((start_step_factor=ini->getd("Walking Config","start_step_factor",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config start_step_factor, usando valor padrão"<<std::endl;
			start_step_factor=0;
		}
	}
	if((p_gain=ini->getd(a,"p_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" p_gain"<<std::endl;
		if((p_gain=ini->getd("Walking Config","p_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config p_gain, usando valor padrão"<<std::endl;
			p_gain=0;
		}
	}
			
	if((i_gain=ini->getd(a,"i_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" i_gain"<<std::endl;
		if((i_gain=ini->getd("Walking Config","i_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config i_gain, usando valor padrão"<<std::endl;
			i_gain=0;
		}
	}
			
	if((d_gain=ini->getd(a,"d_gain",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" d_gain"<<std::endl;
		if((d_gain=ini->getd("Walking Config","d_gain",-1024))==-1024){
			std::cout<<"Erro na leitura do conf.ini section:Walking Config d_gain, usando valor padrão"<<std::endl;
			d_gain=0;
		}
	}
	
	if((turn_angle=ini->getd(a,"turn_angle",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" turn_angle, usando valor padrão 0"<<std::endl;
		turn_angle=0;
	}
	
	if((walk_foward=ini->getd(a,"walk_foward",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" walk_foward, usando valor padrão 0"<<std::endl;
		walk_foward=0;
	}
	
	if((sidle=ini->getd(a,"sidle",-1024))==-1024){
		std::cout<<"Erro na leitura do conf.ini section:"<<a<<" sidle, usando valor padrão 0"<<std::endl;
		sidle=0;
	}

}

void ReadConfig::changeParam(Walking* walk)
{
	walk->X_OFFSET = x_offset;
	walk->Y_OFFSET = y_offset;
	walk->Z_OFFSET = z_offset;
	walk->R_OFFSET = roll_offset;
	walk->P_OFFSET = pitch_offset;
	walk->A_OFFSET = yaw_offset;
	walk->HIP_PITCH_OFFSET = hip_pitch_offset;
	walk->PERIOD_TIME = period_time;
	walk->DSP_RATIO = dsp_ratio;
	walk->STEP_FB_RATIO = step_forward_back_ratio;
	walk->Z_MOVE_AMPLITUDE = foot_height;
	walk->Y_SWAP_AMPLITUDE = swing_right_left;
	walk->Z_SWAP_AMPLITUDE = swing_top_down;
	walk->PELVIS_OFFSET = pelvis_offset;
	walk->ARM_SWING_GAIN = arm_swing_gain;
	walk->BALANCE_KNEE_GAIN = balance_knee_gain;
	walk->BALANCE_ANKLE_PITCH_GAIN = balance_ankle_pitch_gain;
	walk->BALANCE_HIP_ROLL_GAIN = balance_hip_roll_gain;
	walk->BALANCE_ANKLE_ROLL_GAIN = balance_ankle_roll_gain;
	walk->BALANCE_ANGLE_SMOOTH_GAIN = balance_angle_smooth_gain;
	walk->BALANCE_ANGLE_GAIN = balance_angle_gain;
	walk->LEAN_FB = lean_fb_gain;
	walk->LEAN_FB_ACCEL = lean_fb_accel_gain;
	walk->LEAN_TURN = lean_turn_gain;
	walk->START_STEP_FACTOR = start_step_factor;
	
	walk->P_GAIN = p_gain;
	walk->I_GAIN = i_gain;
	walk->D_GAIN = d_gain;
	
	walk->update_param_time();
	walk->update_param_balance();
	walk->update_param_move();
}
