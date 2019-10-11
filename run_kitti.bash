
for seq in  04 06 07 08 09 10
do
    echo ""
    echo ""
    echo ""
    echo "processing $seq"
    data_path="/home/rzh/datasets/kitti/sequences"
    #gdb --args
    ./build/bin/dso_dataset \
    files=${data_path}/$seq/ \
    calib=${data_path}/$seq/camera.txt \
    preset=0 \
    mode=1

   sleep 3
   mv /home/sunny/dso_result.txt kitti_results/$seq.txt
done
