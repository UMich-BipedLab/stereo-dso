
mkdir -p cvo_points
mkdir -p cvo_points_pcd

mkdir -p /home/rayzhang/seagate_2t/kitti/$1/cvo_points 
mkdir -p /home/rayzhang/seagate_2t/kitti/$1/cvo_points_pcd 

./build/bin/pcd_gen \
    files="/home/rayzhang/seagate_2t/kitti/$1/" \
    calib="/home/rayzhang/seagate_2t/kitti/$1/camera.txt" \
    preset=0 \
    mode=1

mv cvo_points/* /home/rayzhang/seagate_2t/kitti/$1/cvo_points/ 
mv cvo_points_pcd/* /home/rayzhang/seagate_2t/kitti/$1/cvo_points_pcd/

echo ""
echo "finish generating cvoTrackingPoints for sequence $1"
echo ""
