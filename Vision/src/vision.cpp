/*--------------------------------------------------------------------
******************************************************************************
  * @file         vision.c
  * @author       Claudio Vilao - ROBOFEI-HT - FEI
  * @version      V0.0.1
  * @created      24/04/2014
  * @Modified by  Isaac Jesus da Silva - ROBOFEI-HT - FEI
  * @Modified     28/09/2016
  * @e-mail       isaac25silva@yahoo.com.br
  * @brief        Vision
  ****************************************************************************
/--------------------------------------------------------------------*/

#include"vision.h"

//using namespace std;

extern "C" bool initServo(int pos1, int pos2)
{
	int pos_servo1 = pos1;
	int pos_servo2 = pos2;
	//int index = 0;
    int deviceIndex = 0; //endereça a USB
    int baudnum = DEFAULT_BAUDNUM; //velocidade de transmissao
	// ---- Open USBDynamixel -----------------------------------------------{
    bool servoComunica = false;

	if( dxl_initialize(deviceIndex, baudnum) == 0 )
	{
		printf( "Failed to open servo!\n");
		printf("\e[1;31mNão há nenhuma placa USB/RS-485 conectada no computador.\n\n\e[0m");
		return true;
	}
	else
	{
		servoComunica = true;
		printf( "Succeed to open Servo!\n");
		if(dxl_read_byte( HEAD_TILT, 3 ) == HEAD_TILT)
		{
		   	printf("Connected and communicating with the head of the robot!\n");
			dxl_write_word(19, 30, pos1);
			dxl_write_word(20, 30, pos2);
			return false;
		}
		else
		{
			printf("\e[0;31mConectou-se a placa USB/RS-485 mas não conseguiu se comunicar com o servo.\e[0m\n");
			printf("\e[0;36mVerifique se a chave que liga os servos motores está na posição ligada.\n\n\e[0m");
		}
	}
	//-----------------------------------------------------------------------------}
}

extern "C" int dxlReadByte(int ID, int Pos)
{
	return dxl_read_byte(ID, Pos);
}

extern "C" int dxlReadWord(int ID, int Pos)
{
	return dxl_read_word(ID, Pos);
}

extern "C" void dxlWriteByte(int ID, int Pos, int value)
{
	dxl_write_byte(ID, Pos, value);
}

extern "C" void dxlWriteWord(int ID, int Pos, int value)
{
	dxl_write_word(ID, Pos, value);
}




