# Outdoor Direct CVO

## Installation
1. Please follow https://github.com/JakobEngel/dso for installing the dependencies
2. Compile: 
In release mode with Intel compiler:
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc
make -j
```
When debugging, please use g++ or clang, and change the CMakeLists.txt CXX flags correspondingly.

## Dataset Format
Place the `camera.txt` into the folder of the dataset. The dataset should contain `image_2` (left camera images), `image_3` (right camera images), `camera.txt` (geometric calibrations).
#### Geometric Calibration File for Pre-Rectified Images: `camera.txt`
```
fx fy cx cy 0.0
[raw_input_width] [raw_input_height]
crop
[cropped_width] [cropped_height]
baseline
```
Example on Kitti Seq05:
```
707.0912 707.0912 601.8873 183.1104 0.0
1226 370
crop
1200 320
0.54
```
Example on Kitti Seq01:
```
718.856 718.856 607.1928 185.2157 0.0
1241 376
crop
1200 320
0.54
```

## Run Experiments on datasets
#### Generating Cvo points
Look at `gen_kitti_pcd.bash`. Replace the data directories.

Outputs (both will lie in the original data folder):
* `cvo_points` is the output directory of cvo points, and can be read by `cvo_test` to do frame to frame alignment test. 
* `cvo_points_pcd` is the corresponding `pcl::PointXYZRGB` pcd files.  We can use `pcl_viewer` to visualize the colored pointcloud.
#### Running Frame to Frame cvo point alignment
Look at `run_pcd_gen_cvo_align.bash` for reference. After we have the cvo points, use the `cvo_test` to run frame to frame experiment. 

The arguments of the program is:
```
 ./build/bin/cvo_test \
        /home/rayzhang/seagate_2t/kitti/$seq/cvo_points \    # place of cvo_points
	kitti_${seq}_out.txt \                               # output trajectory file
	200                                                  # number of frames you want to try
```

#### Running DSO framework with cvo frame alignment
Look at `kitti_o3.bash` for reference. 
