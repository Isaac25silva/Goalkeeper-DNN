sudo apt install libboost-all-dev
sudo apt install libncurses5-dev

cd ./Control/Linux/build_Framework
make all
cd ../../../

cd ./IMU/serial
mkdir build
cd build
cmake ..
sudo make install

cd ../../../

mkdir build
cd build
cmake ../
make all
make install
