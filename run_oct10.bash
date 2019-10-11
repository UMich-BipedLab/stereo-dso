#!/bin/bash

for seq in 04 00 
do
    echo "bash gen_kitti_pcd.bash $seq"
    bash gen_kitti_pcd.bash $seq
    sleep 10
    echo "./build/bin/cvo_test /home/rayzhang/seagate_2t/kitti/$seq/cvo_points kitti_${seq}_out.txt 2000"
    ./build/bin/cvo_test /home/rayzhang/seagate_2t/kitti/$seq/cvo_points kitti_${seq}_out.txt 2000
done
