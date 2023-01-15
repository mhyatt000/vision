#!/bin/bash

git add -A ; git commit -m 'deploy' ; git push ;

clear;
ssh $NODE1 ~/cs/vision/node.sh 0 2 $1 && \ 
    ssh $NODE2 ~/cs/vision/node.sh 1 6 $1 ;

# ssh $NODE1 ~/cs/vision/run.sh && ssh $NODE2 ~/cs/vision/run.sh;

