#!/bin/bash

git add -A ; git commit -m 'deploy' ; git push ;

NODES=("$NODE1" "$NODE2")
for NODE in "${NODES[@]}"
do
    echo $NODE
    ssh $NODE eval "cd cs/vision && git pull"
done

clear;
ssh $NODE1 ~/cs/vision/node.sh && ssh $NODE2 ~/cs/vision/node.sh;

# ssh $NODE1 ~/cs/vision/run.sh && ssh $NODE2 ~/cs/vision/run.sh;

