RoboFEI-HT
=================

This folder have a source code of control body robot, the code was based from robot DARwIn-OP and Jimmy. 

All code within project is GPL GNU v3.

Original source code can be found here:  
https://sourceforge.net/projects/darwinop/  
https://github.com/21stCenturyRobot/HROS5-Framework


Notes
==================
* Operating systems supported & verified: Ubuntu 14.04 LTS
* Robot MUST be in stand up position when launching control process. 
* Currently action_editor is the only way to create motion pages/files.
* ServoTool.py is a alternative to setting the dynamixel servo motors instead use RoboPlus.
* controlRobot.py can be used to control the robot using a graphic interface.


Ubuntu Build Notes
==================
apt-get should have these

Core library dependencies:
build-essential libncurses5-dev libjpeg-dev mplayer mplayer-skins 

PS3/Bluetooth dependencies:
bluez-utils bluez-compat bluez-hcidump libusb-dev libbluetooth-dev joystick


Make Instructions
=================
'make all' in /Linux/build_Framework to compile core framework libraries.

'make clean && make all' in /Linux/build_Framework any time changes are made to core libraries.

action_editor & control are found in /Linux/project

ServoTool.py are found in /Linux/ServoTool-master/src

controlRobot.py are found in /GUIcontrol



