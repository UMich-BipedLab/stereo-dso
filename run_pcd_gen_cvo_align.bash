#!/bin/bash

for seq in 03 00 06 07 08 09 10 
do
    # generating cvo points from images
    echo "bash gen_kitti_pcd.bash $seq"
    bash gen_kitti_pcd.bash $seq
    sleep 10

    # run frame to frame cvo alignment
    echo "./build/bin/cvo_test /home/rzh/datasets/kitti/sequences/$seq/cvo_points kitti_${seq}_out.txt 2000"
    ./build/bin/cvo_test /home/rzh/datasets/kitti/sequences/$seq/cvo_points kitti_${seq}_out.txt 2000
done
