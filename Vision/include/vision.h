#include <iostream>
#include "blackboard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <ctype.h>
#include <string>
#include "dynamixel.h"
#include <unistd.h>

#define DEFAULT_BAUDNUM		1 // 1Mbps
#define HEAD_TILT 20

using namespace std;

extern "C"{

bool initServo(int pos1, int pos2);

int dxlReadByte(int ID, int Pos);

int dxlReadWord(int ID, int Pos);

void dxlWriteByte(int ID, int Pos, int value);

void dxlWriteWord(int ID, int Pos, int value);

}



