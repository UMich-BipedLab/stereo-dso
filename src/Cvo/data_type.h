/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   data_type.h
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Data type definition
 *  @date   August 4, 2019
 **/
#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>
#include<Eigen/StdVector>
#include <opencv2/opencv.hpp>
#include <tbb/concurrent_vector.h>

#include "util/settings.h"
//#define PYR_LEVELS 3

namespace cvo{

  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> cloud_t;
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_row;
  
  struct frame{

    int frame_id;

    int h;        // height of the image without downsampling
    int w;        // width of the image without downsampling

    cv::Mat image;
    cv::Mat intensity;
    cv::Mat depth;
    cv::Mat semantic_img;
    MatrixXf_row semantic_labels;


    Eigen::Vector3f* dI;    // flattened image gradient, (w*h,3). 0: magnitude, 1: dx, 2: dy
    Eigen::Vector3f* dI_pyr[PYR_LEVELS];  // pyramid for dI. dI_pyr[0] = dI
    float* abs_squared_grad[PYR_LEVELS];  // pyramid for absolute squared gradient (dx^2+dy^2)

  };

  struct point_cloud{

    int num_points;

    //typedef std::vector<Eigen::Vector3f> cloud_t;
    cloud_t positions;  // points position. x,y,z
    //Eigen::Matrix<float, Eigen::Dynamic, 5> features;   // features are rgb dx dy
    Eigen::Matrix<float, Eigen::Dynamic, 3> RGB;   // rgb
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> labels; // number of points by number of classes  
    // 0. building 1. sky 2. road
    // 3. vegetation 4. sidewalk 5. car 6. pedestrian
    // 7. cyclist 8. signate 9. fence 10. pole

  };



}

#endif // DATA_TYPE_H
