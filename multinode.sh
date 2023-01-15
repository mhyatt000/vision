#!/bin/bash

git add -A ; git commit -m 'deploy' ; git push ;

clear;
ssh $NODE1 ~/cs/vision/node.sh && \ 
    ssh $NODE2 ~/cs/vision/node.sh;

# ssh $NODE1 ~/cs/vision/run.sh && ssh $NODE2 ~/cs/vision/run.sh;

