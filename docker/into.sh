#!/bin/bash

docker exec --user docker_diffdepth -it ${USER}_diffdepth \
    /bin/bash -c "cd /home/docker_diffdepth; echo ${USER}_diffdepth container; echo ; /bin/bash"