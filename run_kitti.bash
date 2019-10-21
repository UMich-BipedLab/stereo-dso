for seq in  05 00 
do
    echo ""
    echo ""
    echo "==================================================="
    echo "processing $seq"
    data_path="/home/rzh/datasets/kitti/sequences"
    #gdb -ex run --args \
    ./build/bin/dso_dataset \
    files=${data_path}/$seq/ \
    calib=${data_path}/$seq/camera.txt \
    preset=0 \
    mode=1

   sleep 3
   mv /home/rzh/dso_result.txt kitti_results/$seq.txt
   sleep 3
done
