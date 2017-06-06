# Goalkeeper-DNN


### Setup

1. compile the code of the robot running *./setup.sh* in RoboFEI-DNN directory.

## OS and dependencies.

This program was tested in Ubuntu 14.04 LTS 64 bits

* Main Dependencies:
    * cmake
    * g++
    * python 2.7 
    * python-numpy
    * python-opencv
    * [Caffe](https://github.com/NVIDIA/caffe) 

 
 
 ### Downloads
 
    * DNNs trained are available in [link-Networks](https://feiedu-my.sharepoint.com/personal/isaacjesus_fei_edu_br/_layouts/15/guestaccess.aspx?folderid=054e36743516745db8ab64e9c61b71467&authkey=ASlbVagsc7eXeiq4_do7w0w)
 
    * Datasets are available in [link-Dataset](https://feiedu-my.sharepoint.com/personal/isaacjesus_fei_edu_br/_layouts/15/guestaccess.aspx?folderid=0b9239488b2ab4025954c97835f9b22ba&authkey=AZH1KhME-IMzkzpiU2othmg)
    
 
 
  ### Running the test
  
 
 Use the scripts decisionDNN.py to test the DNN and run the control process to receive the decision.
 
 * Running the decision.py:
      * cd Vision/src/
      * python decisionDNN.py ./nets/dataset_googlenet.tar.gz --ws
 
 * Running the control process:
      * ./build/Control/control --p

