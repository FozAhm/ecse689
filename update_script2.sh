#!/bin/bash
git clone https://github.com/FozAhm/ecse689.git /home/ubuntu/ecse689
sudo chown -R ubuntu:ubuntu /home/ubuntu/ecse689/
chmod u+x /home/ubuntu/ecse689/update_script.sh
sudo -u ubuntu /bin/bash /home/ubuntu/ecse689/update_script.sh