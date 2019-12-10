#!/bin/bash
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y
sudo apt install -y python3-pip
pip3 install awscli --upgrade --user
export PATH=$PATH:/home/ubuntu/.local/bin >> ~/.profile
source ~/.profile
pip3 install matplotlib
pip3 install numpy
pip3 install scipy
pip3 install sklearn
ssh-keygen -t rsa -b 4096 -q -N "" -f /home/ubuntu/.ssh/id_rsa
#cat .ssh/id_rsa.pub
sudo reboot