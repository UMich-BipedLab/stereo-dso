
# Input Argument: the sequence number of kitti
# Example: bash gen_kitti_pcd.bash 05

mkdir -p cvo_points
mkdir -p cvo_points_pcd

mkdir -p /home/rzh/datasets/kitti/sequences/$1/cvo_points 
mkdir -p /home/rzh/datasets/kitti/sequences/$1/cvo_points_pcd 

./build/bin/pcd_gen \
    files="/home/rzh/datasets/kitti/sequences/$1/" \
    calib="/home/rzh/datasets/kitti/sequences/$1/camera.txt" \
    preset=0 \
    mode=1

mv cvo_points/* /home/rzh/datasets/kitti/sequences/$1/cvo_points/ 
mv cvo_points_pcd/* /home/rzh/datasets/kitti/sequences/$1/cvo_points_pcd/

echo ""
echo "finish generating cvoTrackingPoints for sequence $1"
echo ""
